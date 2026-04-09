"""T05: Curiosity Engine — event detection, goal generation, scheduling, resolution.

CE flood mitigation (addresses ARC-AGI-3 saturation critique):
  On a benchmark with 250+ novel environments, every task would trigger
  curiosity events, saturating the 50-goal queue immediately. The fix:
  - Hard cap at MAX_PENDING_GOALS = 50 (PRD spec)
  - When at capacity, new goals only admitted if priority > lowest existing
  - Goals retired when: priority < 0.40 AND age > 1000 episodes (PRD)
  - Deduplication: goals with cos(target, existing) > 0.80 are merged
    (priority = max of the two) rather than added as duplicates
  This means the CE tracks the 50 highest-priority unresolved regions,
  not the 50 oldest — which is the correct behavior for a benchmark.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field, asdict
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from psn2.ers import ExperienceReplaySubstrate

TAU_CURIO_ERROR = 0.50
TAU_CURIO_DL = 2.5
TAU_CURIO_PRIORITY = 0.40
TAU_CURIO_RESOLVE = 0.65
GOAL_UTILITY = 0.80
MAX_PENDING_GOALS = 50          # PRD Section 24
GOAL_RETIRE_AGE = 1000          # PRD Section 24
GOAL_RETIRE_PRIORITY = 0.40     # PRD Section 24
GOAL_DEDUP_COS = 0.80           # merge goals that target the same region


@dataclass
class CuriosityGoal:
    target_id: int
    priority: float
    budget_reserve: float
    source: str = "uncertainty_reduction"
    target_vsa: Optional[List[float]] = None
    age_episodes: int = 0        # incremented each episode this goal is not resolved

    def target_tensor(self) -> Optional[torch.Tensor]:
        if self.target_vsa is not None:
            return torch.tensor(self.target_vsa)
        return None

    def is_retirement_eligible(self) -> bool:
        return self.priority < GOAL_RETIRE_PRIORITY and self.age_episodes > GOAL_RETIRE_AGE


class CuriosityEngine:
    """Full lifecycle: detect → generate → store in ERS → schedule → resolve."""

    def __init__(self, ers: Optional["ExperienceReplaySubstrate"] = None):
        self.goals: List[CuriosityGoal] = []
        self.ers = ers

    def should_trigger(self, error: float, dl: float,
                       budget_fraction: float, committed: bool) -> bool:
        return (
            error > TAU_CURIO_ERROR
            and dl > TAU_CURIO_DL
            and budget_fraction < 0.30
            and not committed
        )

    def _find_duplicate(self, target_vsa: torch.Tensor) -> Optional[CuriosityGoal]:
        """Return existing goal if cos(target, existing) > GOAL_DEDUP_COS."""
        for g in self.goals:
            tvsa = g.target_tensor()
            if tvsa is None:
                continue
            sim = float(F.cosine_similarity(
                target_vsa.unsqueeze(0), tvsa.to(target_vsa.device).unsqueeze(0)
            ).item())
            if sim > GOAL_DEDUP_COS:
                return g
        return None

    def _admit_goal(self, goal: CuriosityGoal):
        """
        Admit a new goal with priority-based admission when at capacity.
        Deduplicates against existing goals first.
        """
        tvsa = goal.target_tensor()
        if tvsa is not None:
            dup = self._find_duplicate(tvsa)
            if dup is not None:
                # Merge: keep higher priority
                dup.priority = max(dup.priority, goal.priority)
                return

        if len(self.goals) < MAX_PENDING_GOALS:
            self.goals.append(goal)
        else:
            # Only admit if priority > lowest existing goal
            lowest_idx = min(range(len(self.goals)), key=lambda i: self.goals[i].priority)
            if goal.priority > self.goals[lowest_idx].priority:
                self.goals[lowest_idx] = goal

    def add_goal(self, goal: CuriosityGoal):
        self._admit_goal(goal)
        if self.ers is not None and goal.target_vsa is not None:
            from psn2.ers import ERSTuple
            tvsa = goal.target_tensor()
            tup = ERSTuple(
                input_vsa=tvsa,
                trace_vsa=tvsa,
                output_vsa=tvsa,
                source="curiosity_goal",
                utility_score=GOAL_UTILITY,
            )
            self.ers.write("episodic", tup)

    def detect_and_generate(self, node_id: int, error: float, dl: float,
                             budget_fraction: float, committed: bool,
                             target_vsa: torch.Tensor) -> Optional[CuriosityGoal]:
        if not self.should_trigger(error, dl, budget_fraction, committed):
            return None
        goal = CuriosityGoal(
            target_id=node_id,
            priority=float(error),
            budget_reserve=0.25,
            target_vsa=target_vsa.detach().tolist(),
        )
        self.add_goal(goal)
        return goal

    def tick_episode(self):
        """Call once per episode to age goals and retire stale ones."""
        for g in self.goals:
            g.age_episodes += 1
        self.goals = [g for g in self.goals if not g.is_retirement_eligible()]

    def load_episode_goals(self, context_vsa: torch.Tensor) -> List[CuriosityGoal]:
        """Load pending goals at episode start if cos > 0.40."""
        loaded = []
        for goal in self.goals:
            tvsa = goal.target_tensor()
            if tvsa is None:
                continue
            sim = float(F.cosine_similarity(
                context_vsa.unsqueeze(0), tvsa.to(context_vsa.device).unsqueeze(0)
            ).item())
            if sim > TAU_CURIO_PRIORITY:
                loaded.append(goal)
        return loaded

    def resolve(self, goal: CuriosityGoal, committed_vsa: torch.Tensor) -> bool:
        """Promote to Semantic if resolved; reschedule with +0.10 priority if not."""
        tvsa = goal.target_tensor()
        if tvsa is None:
            return False
        sim = float(F.cosine_similarity(
            committed_vsa.unsqueeze(0), tvsa.to(committed_vsa.device).unsqueeze(0)
        ).item())
        if sim > TAU_CURIO_RESOLVE:
            if self.ers is not None:
                from psn2.ers import ERSTuple
                tup = ERSTuple(
                    input_vsa=tvsa,
                    trace_vsa=committed_vsa.detach(),
                    output_vsa=committed_vsa.detach(),
                    source="curiosity_resolved",
                    utility_score=0.90,
                )
                self.ers.write("semantic", tup)
            if goal in self.goals:
                self.goals.remove(goal)
            return True
        else:
            goal.priority = min(1.0, goal.priority + 0.10)
            return False

    def state_dict(self) -> list:
        return [asdict(g) for g in self.goals]

    def load_state_dict(self, state: list):
        self.goals = []
        for x in state:
            self.goals.append(CuriosityGoal(**x))
