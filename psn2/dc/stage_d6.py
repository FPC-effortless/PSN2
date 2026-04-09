"""Developmental Stage D6 — Full Integration and Meta-Learning.

PRD gates (Section 16.6):
  sample_complexity_ratio >= 10.0  (on 3/5 families)
  few_shot_k1_efficiency >= 0.40
  few_shot_k5_efficiency >= 0.70
  compositional_split_score >= 0.75
  growth_ledger_i24_satisfied == True
  anti_forgetting_regression <= 0.02
  human_parity_profile_pass == True
Requires D5 certified.
"""
from __future__ import annotations

from psn2.dc.gate_certifier import GateCertifier
from psn2.dc.stage_d5 import StageD5


class StageD6:
    """D6: Full Integration and Meta-Learning — continuous operational phase."""

    STAGE_NAME = "D6"

    def __init__(self, d5: StageD5):
        self.d5 = d5
        self._scr = 0.0
        self._few_shot_k1 = 0.0
        self._few_shot_k5 = 0.0
        self._comp_split = 0.0
        self._i24_satisfied = False
        self._anti_forgetting = 1.0   # regression fraction; must be <= 0.02
        self._human_parity = False

        self.certifier = GateCertifier(
            stage_name=self.STAGE_NAME,
            gates={
                "sample_complexity_ratio":    lambda: self._scr,
                "few_shot_k1_efficiency":     lambda: self._few_shot_k1,
                "few_shot_k5_efficiency":     lambda: self._few_shot_k5,
                "compositional_split_score":  lambda: self._comp_split,
                "growth_ledger_i24":          lambda: float(self._i24_satisfied),
                "anti_forgetting_regression": lambda: self._anti_forgetting,
                "human_parity_profile":       lambda: float(self._human_parity),
            },
            thresholds={
                "sample_complexity_ratio":    10.0,
                "few_shot_k1_efficiency":     0.40,
                "few_shot_k5_efficiency":     0.70,
                "compositional_split_score":  0.75,
                "growth_ledger_i24":          1.0,   # must be True (1.0)
                "anti_forgetting_regression": 0.02,
                "human_parity_profile":       1.0,   # must be True (1.0)
            },
            comparators={
                "sample_complexity_ratio":    ">=",
                "few_shot_k1_efficiency":     ">=",
                "few_shot_k5_efficiency":     ">=",
                "compositional_split_score":  ">=",
                "growth_ledger_i24":          ">=",
                "anti_forgetting_regression": "<=",
                "human_parity_profile":       ">=",
            },
        )

    def update_metrics(self, scr: float, few_shot_k1: float, few_shot_k5: float,
                       comp_split: float, i24_satisfied: bool,
                       anti_forgetting: float, human_parity: bool):
        self._scr = scr
        self._few_shot_k1 = few_shot_k1
        self._few_shot_k5 = few_shot_k5
        self._comp_split = comp_split
        self._i24_satisfied = i24_satisfied
        self._anti_forgetting = anti_forgetting
        self._human_parity = human_parity

    def is_complete(self) -> bool:
        return self.d5.is_complete() and self.certifier.is_certified()

    def report(self) -> str:
        d5_ok = "PASS" if self.d5.is_complete() else "FAIL"
        return f"D5 prerequisite: [{d5_ok}]\n" + self.certifier.report()
