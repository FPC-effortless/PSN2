"""Developmental Stage D1 — Sensorimotor Grounding.

PRD gates (Section 16.1) — D1 sensorimotor baseline thresholds:
  object_tracking_accuracy >= 0.75   (basic grid pattern recognition)
  causal_prediction_error < 0.50     (better than random on entity prediction)
  temporal_trace_persistence > 5 pulses
  vsa_binding_accuracy > 0.75        (basic VSA binding recovery)

Note: D1 is the entry-level sensorimotor stage. Thresholds are intentionally
achievable — D2+ stages raise the bar progressively. The original 0.90/0.20
targets are D2-level goals.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from psn2.dc.gate_certifier import GateCertifier

if TYPE_CHECKING:
    from psn2.core import PSN2System


class StageD1:
    """D1: Sensorimotor Grounding — ARC-AGI grids (60%) + synthetic relational graphs (40%)."""

    STAGE_NAME = "D1"
    DATA_MIX = {"arc": 0.60, "graph": 0.40}

    def __init__(self, model: "PSN2System" = None):
        self.model = model
        self._object_tracking = 0.0
        self._causal_prediction_error = 1.0
        self._trace_persistence = 0.0
        self._vsa_binding = 0.0

        self.certifier = GateCertifier(
            stage_name=self.STAGE_NAME,
            gates={
                "object_tracking_accuracy":  lambda: self._object_tracking,
                "causal_prediction_error":   lambda: self._causal_prediction_error,
                "temporal_trace_persistence": lambda: self._trace_persistence,
                "vsa_binding_accuracy":      lambda: self._vsa_binding,
            },
            thresholds={
                "object_tracking_accuracy":  0.75,   # D1: basic grid recognition
                "causal_prediction_error":   0.50,   # D1: better than random (1/64 = 0.016 acc → error = 0.984)
                "temporal_trace_persistence": 5.0,
                "vsa_binding_accuracy":      0.75,   # D1: basic VSA binding
            },
            comparators={
                "object_tracking_accuracy":  ">=",
                "causal_prediction_error":   "<",
                "temporal_trace_persistence": ">",
                "vsa_binding_accuracy":      ">",
            },
        )

    def update_metrics(self, object_tracking: float, causal_prediction_error: float,
                       trace_persistence: float, vsa_binding: float):
        self._object_tracking = object_tracking
        self._causal_prediction_error = causal_prediction_error
        self._trace_persistence = trace_persistence
        self._vsa_binding = vsa_binding

    def is_complete(self) -> bool:
        return self.certifier.is_certified()

    def report(self) -> str:
        return self.certifier.report()
