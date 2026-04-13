"""Developmental Stage D5 — Abstract Reasoning and Formal Competence.

PRD gates (Section 16.5):
  arc_agi_improvement > 0.0  (improvement over D4 baseline)
  math_verification_rate >= 0.90
  multi_step_planning_success >= 0.80
  compositional_split_score >= 0.75
Requires D4 certified.
"""
from __future__ import annotations

from psn2.dc.gate_certifier import GateCertifier
from psn2.dc.stage_d4 import StageD4


class StageD5:
    """D5: Abstract Reasoning and Formal Competence."""

    STAGE_NAME = "D5"

    def __init__(self, d4: StageD4):
        self.d4 = d4
        self._arc_improvement = 0.0
        self._math_verify = 0.0
        self._planning = 0.0
        self._comp_split = 0.0

        self.certifier = GateCertifier(
            stage_name=self.STAGE_NAME,
            gates={
                "arc_agi_improvement":          lambda: self._arc_improvement,
                "math_verification_rate":       lambda: self._math_verify,
                "multi_step_planning_success":  lambda: self._planning,
                "compositional_split_score":    lambda: self._comp_split,
            },
            thresholds={
                "arc_agi_improvement":          0.0,
                "math_verification_rate":       0.90,
                "multi_step_planning_success":  0.80,
                "compositional_split_score":    0.75,
            },
            comparators={
                "arc_agi_improvement":          ">",
                "math_verification_rate":       ">=",
                "multi_step_planning_success":  ">=",
                "compositional_split_score":    ">=",
            },
        )

    def update_metrics(self, arc_improvement: float, math_verify: float,
                       planning: float, comp_split: float):
        self._arc_improvement = arc_improvement
        self._math_verify = math_verify
        self._planning = planning
        self._comp_split = comp_split

    def is_complete(self) -> bool:
        return self.d4.is_complete() and self.certifier.is_certified()

    def report(self) -> str:
        d4_ok = "PASS" if self.d4.is_complete() else "FAIL"
        return f"D4 prerequisite: [{d4_ok}]\n" + self.certifier.report()
