"""T23: Trace Reconstructor — observed (input, steps, output) → synthetic ERS tuple."""
from __future__ import annotations

import torch
from typing import List, TYPE_CHECKING

from psn2.vsa import normalize, bundle

if TYPE_CHECKING:
    from psn2.ers import ExperienceReplaySubstrate, ERSTuple
    from psn2.sls.social_shape import SocialShape


class TraceReconstructor:
    """
    Reconstructs a synthetic ERS tuple from an observed agent trace.
    Observed trace: (input_vsa, step_vsas, output_vsa)
    """

    def __init__(self, ers: "ExperienceReplaySubstrate"):
        self.ers = ers

    def reconstruct(self, social_shape: "SocialShape",
                    input_vsa: torch.Tensor,
                    step_vsas: List[torch.Tensor],
                    output_vsa: torch.Tensor) -> "ERSTuple":
        """
        Encode observed trace into a synthetic ERS tuple.
        Trust < 0.30: goes to Working tier only.
        """
        from psn2.ers import ERSTuple

        # Bundle step VSAs into a trace representation
        if step_vsas:
            trace_vsa = normalize(bundle(step_vsas))
        else:
            trace_vsa = normalize(input_vsa + output_vsa)

        utility = social_shape.trust * 0.8  # trust-weighted utility

        tup = ERSTuple(
            input_vsa=input_vsa.detach(),
            trace_vsa=trace_vsa.detach(),
            output_vsa=output_vsa.detach(),
            source=f"social:{social_shape.agent_id}",
            utility_score=float(utility),
        )

        tier = "working" if social_shape.is_suspicious else "episodic"
        self.ers.write(tier, tup)
        return tup
