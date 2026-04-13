"""T25: Goal Generator — creates CuriosityGoal shapes and stores in ERS."""
from __future__ import annotations

import torch
from dataclasses import dataclass, asdict
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from psn2.ers import ExperienceReplaySubstrate


@dataclass
class CuriosityGoalShape:
    shape_class: str = "goal"
    goal_type: str = "uncertainty_reduction"
    node_id: int = 0
    target_vsa: torch.Tensor = None
    priority: float = 0.5
    budget_reserve: float = 0.25

    def __post_init__(self):
        if self.target_vsa is None:
            raise ValueError("target_vsa required")


class GoalGenerator:
    """Generates CuriosityGoal shapes and stores them in ERS."""

    GOAL_UTILITY = 0.80

    def __init__(self, ers: "ExperienceReplaySubstrate"):
        self.ers = ers
        self.goals: List[CuriosityGoalShape] = []

    def generate(self, node_id: int, target_vsa: torch.Tensor,
                 priority: float = 0.5) -> CuriosityGoalShape:
        goal = CuriosityGoalShape(
            node_id=node_id,
            target_vsa=target_vsa.detach(),
            priority=priority,
            budget_reserve=0.25,
        )
        self.goals.append(goal)

        # Store in ERS with utility_score=0.80
        from psn2.ers import ERSTuple
        tup = ERSTuple(
            input_vsa=target_vsa.detach(),
            trace_vsa=target_vsa.detach(),
            output_vsa=target_vsa.detach(),
            source="curiosity_goal",
            utility_score=self.GOAL_UTILITY,
        )
        self.ers.write("episodic", tup)
        return goal

    def state_dict(self) -> list:
        return [
            {
                "node_id": g.node_id,
                "target_vsa": g.target_vsa.tolist(),
                "priority": g.priority,
                "budget_reserve": g.budget_reserve,
            }
            for g in self.goals
        ]

    def load_state_dict(self, state: list):
        self.goals = []
        for s in state:
            g = CuriosityGoalShape(
                node_id=s["node_id"],
                target_vsa=torch.tensor(s["target_vsa"]),
                priority=s["priority"],
                budget_reserve=s["budget_reserve"],
            )
            self.goals.append(g)
