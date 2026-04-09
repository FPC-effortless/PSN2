"""T25: Goal Scheduler — load pending goals at episode start, resolve after episode."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Optional, TYPE_CHECKING

from psn2.ce.goal_generator import CuriosityGoalShape

if TYPE_CHECKING:
    from psn2.ers import ExperienceReplaySubstrate

TAU_CURIO_PRIORITY = 0.40   # cos threshold for loading goals
TAU_CURIO_RESOLVE = 0.65    # cos threshold for resolution


class GoalScheduler:
    """
    Scheduling: at episode start, retrieve pending goals from Episodic ERS by cos similarity.
    Resolution: promote to Semantic if resolved; reschedule with +0.10 priority if not.
    """

    def __init__(self, ers: "ExperienceReplaySubstrate", goal_generator: "GoalGenerator"):
        self.ers = ers
        self.goal_generator = goal_generator

    def load_episode_goals(self, context_vsa: torch.Tensor) -> List[CuriosityGoalShape]:
        """
        At episode start: retrieve pending curiosity goals from Episodic ERS.
        Returns goals with cos(context, target) > 0.40.
        """
        candidates = self.ers.retrieve(context_vsa, k=10, goal_type="curiosity_goal")
        loaded = []
        for tup in candidates:
            if tup.source != "curiosity_goal":
                continue
            sim = float(F.cosine_similarity(
                context_vsa.unsqueeze(0), tup.input_vsa.unsqueeze(0)
            ).item())
            if sim > TAU_CURIO_PRIORITY:
                # Reconstruct goal shape
                goal = CuriosityGoalShape(
                    node_id=0,
                    target_vsa=tup.input_vsa,
                    priority=tup.utility_score,
                    budget_reserve=0.25,
                )
                loaded.append(goal)
        return loaded

    def resolve(self, goal: CuriosityGoalShape, committed_vsa: torch.Tensor) -> bool:
        """
        Resolve a goal after episode.
        Returns True if resolved (promotes to Semantic), False if rescheduled.
        """
        sim = float(F.cosine_similarity(
            committed_vsa.unsqueeze(0), goal.target_vsa.unsqueeze(0)
        ).item())

        if sim > TAU_CURIO_RESOLVE:
            # Promote to Semantic
            from psn2.ers import ERSTuple
            tup = ERSTuple(
                input_vsa=goal.target_vsa,
                trace_vsa=committed_vsa.detach(),
                output_vsa=committed_vsa.detach(),
                source="curiosity_resolved",
                utility_score=0.90,
            )
            self.ers.write("semantic", tup)
            return True
        else:
            # Reschedule with +0.10 priority
            goal.priority = min(1.0, goal.priority + 0.10)
            self.goal_generator.generate(goal.node_id, goal.target_vsa, goal.priority)
            return False
