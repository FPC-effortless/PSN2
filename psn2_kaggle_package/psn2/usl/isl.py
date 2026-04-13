"""T22: Inner Speech Loop (ISL) — draft verbal propositions as modulatory evidence."""
from __future__ import annotations

import torch
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from psn2.usl.codec import USLCodec
    from psn2.ers import ExperienceReplaySubstrate, ERSTuple

ISL_TRIGGER_UNCERTAINTY = 0.55
ISL_BUDGET_FRACTION = 0.25


class InnerSpeechLoop:
    """
    Trigger: regime==Recursive AND uncertainty > 0.55 AND pulse_remaining > B_max/4
    Generate draft verbal proposition → re-encode → write to Working ERS as modulatory input.
    ISL output feeds Phase A as modulatory (NOT proximal) input.
    """

    def __init__(self, codec: "USLCodec", ers: "ExperienceReplaySubstrate", dim: int):
        self.codec = codec
        self.ers = ers
        self.dim = dim
        self.last_draft: Optional[torch.Tensor] = None

    def should_trigger(self, regime: str, uncertainty: float,
                       pulse_remaining: int, budget_max: int) -> bool:
        return (
            regime == "recursive"
            and uncertainty > ISL_TRIGGER_UNCERTAINTY
            and pulse_remaining > budget_max * ISL_BUDGET_FRACTION
        )

    def run(self, best_candidate_shape: torch.Tensor,
            pulse_remaining: int, budget_max: int,
            regime: str = "recursive", uncertainty: float = 0.6) -> Optional[torch.Tensor]:
        """
        Returns modulatory_input tensor if triggered, else None.
        The returned tensor feeds into Phase A as modulatory (not proximal) input.
        """
        if not self.should_trigger(regime, uncertainty, pulse_remaining, budget_max):
            return None

        # Generate draft text (token logits)
        token_logits, passed = self.codec.generate(best_candidate_shape, n_tokens=8)
        if not passed:
            return None

        # Re-encode draft
        draft_ids = token_logits.argmax(dim=-1)
        draft_shape, _ = self.codec.understand(draft_ids)
        self.last_draft = draft_shape.detach()

        # Write to Working ERS with source='inner_speech'
        from psn2.ers import ERSTuple
        tup = ERSTuple(
            input_vsa=best_candidate_shape.detach(),
            trace_vsa=draft_shape.detach(),
            output_vsa=draft_shape.detach(),
            source="inner_speech",
            utility_score=0.5,
        )
        self.ers.write("working", tup)

        # Return as modulatory input (attenuated to prevent override)
        return draft_shape.detach() * 0.3
