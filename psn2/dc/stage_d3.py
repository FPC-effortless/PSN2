"""Developmental Stage D3 — Social and Theory-of-Mind Grounding.

PRD gates (Section 16.3):
  goal_inference_accuracy >= 0.75
  false_belief_accuracy >= 0.75
  trust_calibration_rmse < 0.15
  emotional_shape_induction_accuracy >= 0.70
Requires D2 certified.
"""
from __future__ import annotations

from psn2.dc.gate_certifier import GateCertifier
from psn2.dc.stage_d2 import StageD2


class StageD3:
    """D3: Social and Theory-of-Mind Grounding."""

    STAGE_NAME = "D3"

    def __init__(self, d2: StageD2):
        self.d2 = d2
        self._goal_inference = 0.0
        self._false_belief = 0.0
        self._trust_rmse = 1.0
        self._emo_induction = 0.0

        self.certifier = GateCertifier(
            stage_name=self.STAGE_NAME,
            gates={
                "goal_inference_accuracy":           lambda: self._goal_inference,
                "false_belief_accuracy":             lambda: self._false_belief,
                "trust_calibration_rmse":            lambda: self._trust_rmse,
                "emotional_shape_induction_accuracy": lambda: self._emo_induction,
            },
            thresholds={
                "goal_inference_accuracy":           0.75,
                "false_belief_accuracy":             0.75,
                "trust_calibration_rmse":            0.15,
                "emotional_shape_induction_accuracy": 0.70,
            },
            comparators={
                "goal_inference_accuracy":           ">=",
                "false_belief_accuracy":             ">=",
                "trust_calibration_rmse":            "<",
                "emotional_shape_induction_accuracy": ">=",
            },
        )

    def update_metrics(self, goal_inference: float, false_belief: float,
                       trust_rmse: float, emo_induction: float):
        self._goal_inference = goal_inference
        self._false_belief = false_belief
        self._trust_rmse = trust_rmse
        self._emo_induction = emo_induction

    def is_complete(self) -> bool:
        return self.d2.is_complete() and self.certifier.is_certified()

    def report(self) -> str:
        d2_ok = "PASS" if self.d2.is_complete() else "FAIL"
        return f"D2 prerequisite: [{d2_ok}]\n" + self.certifier.report()
