"""Developmental Stage D4 — Linguistic Grounding.

PRD gates (Section 16.4):
  usl_roundtrip_fidelity >= 0.85
  language_grounded_analogy >= 0.80
  isl_coherent_episodes >= 0.70
  linguistic_bond_vsa_recovery >= 0.90
Requires D3 certified.
"""
from __future__ import annotations

from psn2.dc.gate_certifier import GateCertifier
from psn2.dc.stage_d3 import StageD3


class StageD4:
    """D4: Linguistic Grounding — USL codec training, LAL population, ISL activation."""

    STAGE_NAME = "D4"

    def __init__(self, d3: StageD3):
        self.d3 = d3
        self._usl_fidelity = 0.0
        self._lang_analogy = 0.0
        self._isl_coherent = 0.0
        self._ling_bond_recovery = 0.0

        self.certifier = GateCertifier(
            stage_name=self.STAGE_NAME,
            gates={
                "usl_roundtrip_fidelity":       lambda: self._usl_fidelity,
                "language_grounded_analogy":    lambda: self._lang_analogy,
                "isl_coherent_episodes":        lambda: self._isl_coherent,
                "linguistic_bond_vsa_recovery": lambda: self._ling_bond_recovery,
            },
            thresholds={
                "usl_roundtrip_fidelity":       0.85,
                "language_grounded_analogy":    0.80,
                "isl_coherent_episodes":        0.70,
                "linguistic_bond_vsa_recovery": 0.90,
            },
        )

    def update_metrics(self, usl_fidelity: float, lang_analogy: float,
                       isl_coherent: float, ling_bond_recovery: float):
        self._usl_fidelity = usl_fidelity
        self._lang_analogy = lang_analogy
        self._isl_coherent = isl_coherent
        self._ling_bond_recovery = ling_bond_recovery

    def is_complete(self) -> bool:
        return self.d3.is_complete() and self.certifier.is_certified()

    def report(self) -> str:
        d3_ok = "PASS" if self.d3.is_complete() else "FAIL"
        return f"D3 prerequisite: [{d3_ok}]\n" + self.certifier.report()
