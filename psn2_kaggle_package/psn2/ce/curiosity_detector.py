"""T25: Curiosity Detector — all four conditions for curiosity event detection."""
from __future__ import annotations

import torch
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from psn2.ce.goal_generator import GoalGenerator

TAU_CURIO_ERROR = 0.50
TAU_CURIO_DL = 2.5
TAU_CURIO_BUDGET = 0.30  # budget < 30%


class CuriosityDetector:
    """
    Detects curiosity events with all four conditions:
      e > 0.50, dl > 2.5, budget < 30%, node not committed
    """

    def __init__(self, goal_generator: "GoalGenerator"):
        self.goal_generator = goal_generator

    def check_and_generate(self, node_id: int, error: float, dl: float,
                            budget_fraction: float, committed: bool,
                            target_vsa: torch.Tensor) -> bool:
        """
        Returns True if curiosity event detected and goal generated.
        """
        if self._is_curious(error, dl, budget_fraction, committed):
            self.goal_generator.generate(node_id, target_vsa, priority=error)
            return True
        return False

    def _is_curious(self, error: float, dl: float,
                    budget_fraction: float, committed: bool) -> bool:
        return (
            error > TAU_CURIO_ERROR
            and dl > TAU_CURIO_DL
            and budget_fraction < TAU_CURIO_BUDGET
            and not committed
        )
