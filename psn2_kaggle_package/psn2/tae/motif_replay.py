"""T24: Motif Replay — inject pre-formed shapes in Perceptive regime."""
from __future__ import annotations

import torch
from typing import Optional, Callable, TYPE_CHECKING

from psn2.tae.tal import TemporalMotifShape, TemporalAttractorLibrary

if TYPE_CHECKING:
    pass


class MotifReplay:
    """
    Replays a temporal motif by injecting its pulse_sequence_vsa as a
    pre-formed shape in the Perceptive regime.
    Verifier preliminary check is applied before injection.
    """

    def __init__(self, tal: TemporalAttractorLibrary):
        self.tal = tal
        self.active_replay: Optional[TemporalMotifShape] = None
        self.replay_motif_idx: Optional[int] = None

    def attempt_replay(self, trigger_vsa: torch.Tensor,
                       regime: str,
                       verifier_fn: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """
        Returns pre-formed shape tensor if replay is triggered, else None.
        Only activates in Perceptive regime.
        """
        if regime != "perceptive":
            return None

        candidates = self.tal.retrieve(trigger_vsa, k=1)
        if not candidates:
            return None

        sim, motif = candidates[0]
        if not motif.promoted:
            return None  # Only replay promoted motifs

        # Preliminary verifier check
        if verifier_fn is not None:
            if not verifier_fn(motif.pulse_sequence_vsa):
                return None

        self.active_replay = motif
        self.replay_motif_idx = self.tal.motifs.index(motif)
        return motif.pulse_sequence_vsa.clone()

    def record_outcome(self, success: bool, actual_pulses_used: int):
        """Feed back outcome to TAL for success rate and savings tracking."""
        if self.replay_motif_idx is not None and self.active_replay is not None:
            savings = float(self.active_replay.duration - actual_pulses_used)
            self.tal.update_feedback(self.replay_motif_idx, success, savings)
            self.active_replay = None
            self.replay_motif_idx = None
