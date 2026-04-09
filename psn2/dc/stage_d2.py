"""Developmental Stage D2 — Causal and Relational Grounding.

PRD gates (Section 16.2):
  causal_intervention_accuracy >= 0.80
  abstract_analogy_score >= 0.75
  vsa_causal_bond_recall >= 0.90
  compositional_split_score >= 0.65
Requires D1 certified.
"""
from __future__ import annotations

from psn2.dc.gate_certifier import GateCertifier
from psn2.dc.stage_d1 import StageD1


class StageD2:
    """D2: Causal and Relational Grounding."""

    STAGE_NAME = "D2"

    def __init__(self, d1: StageD1):
        self.d1 = d1
        self._causal_acc = 0.0
        self._analogy_score = 0.0
        self._bond_recall = 0.0
        self._comp_split = 0.0

        self.certifier = GateCertifier(
            stage_name=self.STAGE_NAME,
            gates={
                "causal_intervention_accuracy": lambda: self._causal_acc,
                "abstract_analogy_score":       lambda: self._analogy_score,
                "vsa_causal_bond_recall":       lambda: self._bond_recall,
                "compositional_split_score":    lambda: self._comp_split,
            },
            thresholds={
                "causal_intervention_accuracy": 0.80,
                "abstract_analogy_score":       0.75,
                "vsa_causal_bond_recall":       0.90,
                "compositional_split_score":    0.65,
            },
        )

    def update_metrics(self, causal_acc: float, analogy_score: float,
                       bond_recall: float, comp_split: float):
        self._causal_acc = causal_acc
        self._analogy_score = analogy_score
        self._bond_recall = bond_recall
        self._comp_split = comp_split

    def is_complete(self) -> bool:
        return self.d1.is_complete() and self.certifier.is_certified()

    def report(self) -> str:
        d1_ok = "PASS" if self.d1.is_complete() else "FAIL"
        return f"D1 prerequisite: [{d1_ok}]\n" + self.certifier.report()
