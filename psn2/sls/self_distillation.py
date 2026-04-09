"""T23: Self-Distillation — treat own verified traces as high-trust social observations."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psn2.ers import ExperienceReplaySubstrate, ERSTuple


class SelfDistillation:
    """
    Treats the system's own verified traces as high-trust social observations.
    Writes them to Episodic ERS with utility_score=0.90.
    """

    SELF_TRUST = 1.0
    SELF_UTILITY = 0.90

    def __init__(self, ers: "ExperienceReplaySubstrate"):
        self.ers = ers

    def record_verified_trace(self, input_vsa: torch.Tensor,
                               trace_vsa: torch.Tensor,
                               output_vsa: torch.Tensor):
        """Record a self-verified trace as a high-trust episodic memory."""
        from psn2.ers import ERSTuple
        tup = ERSTuple(
            input_vsa=input_vsa.detach(),
            trace_vsa=trace_vsa.detach(),
            output_vsa=output_vsa.detach(),
            source="self_distillation",
            utility_score=self.SELF_UTILITY,
        )
        self.ers.write("episodic", tup)
        return tup
