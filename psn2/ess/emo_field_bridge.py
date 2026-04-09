"""T17: Field Vector Compatibility Bridge — maps legacy F_t dimensions to ESS mechanisms.

Legacy field dimensions (7):
  focus, curiosity, caution, urgency, play, consistency, aesthetic

All legacy code paths are replaced; callers get ESS-mediated behaviour.
"""
from __future__ import annotations

import torch
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from psn2.ess.emotional_shapes import EmotionalShapeSystem
    from psn2.ce.curiosity_detector import CuriosityDetector


class EmoFieldBridge:
    """
    Translates a legacy 7-dim field vector F_t into ESS + CE actions.

    Mapping:
      focus       -> attention-narrowing bond (handled by self-shape)
      curiosity   -> CE goal generation
      caution     -> threat_fear emotional shape
      urgency     -> urgency emotional shape
      play        -> curiosity_drive emotional shape (novelty)
      consistency -> self-shape regulatory bond
      aesthetic   -> aesthetic_resonance emotional shape
    """

    FIELD_DIMS = ["focus", "curiosity", "caution", "urgency", "play", "consistency", "aesthetic"]
    FIELD_TO_EMO = {
        "caution":   "threat_fear",
        "urgency":   "urgency",
        "play":      "curiosity_drive",
        "aesthetic": "aesthetic_resonance",
    }

    def __init__(self, ess: "EmotionalShapeSystem", ce_detector=None):
        self.ess = ess
        self.ce_detector = ce_detector  # optional CuriosityDetector

    def apply(self, field_vec: torch.Tensor, shape_centroid: torch.Tensor,
              node_error: float = 0.0, node_dl: float = 0.0,
              budget_fraction: float = 1.0, node_committed: bool = False):
        """
        Process a legacy 7-dim field vector.
        field_vec: [7] float tensor
        """
        assert field_vec.shape[-1] == 7, "Legacy field vector must be 7-dimensional"
        vals = {dim: float(field_vec[i].item()) for i, dim in enumerate(self.FIELD_DIMS)}

        # Emotional shape induction for mapped dimensions
        for field_dim, emo_type in self.FIELD_TO_EMO.items():
            strength = vals[field_dim]
            if strength > 0.3:
                # Synthesize a semantic vector biased by the field value
                semantic_v = shape_centroid * strength
                from psn2.ess.emotional_shapes import EmotionalShape
                emo = EmotionalShape(emo_type=emo_type, semantic_v=semantic_v.detach(), strength=strength)
                self.ess.active.append(emo)

        # curiosity -> CE goal
        if self.ce_detector is not None and vals["curiosity"] > 0.5:
            self.ce_detector.check_and_generate(
                node_id=0,
                error=node_error,
                dl=node_dl,
                budget_fraction=budget_fraction,
                committed=node_committed,
                target_vsa=shape_centroid,
            )

        # focus and consistency are handled by self-shape (no direct ESS action here)
