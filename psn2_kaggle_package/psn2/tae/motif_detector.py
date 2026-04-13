"""T24: Motif Detector — detect recurring pulse sequences."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from psn2.vsa import normalize, bundle
from psn2.tae.tal import TemporalMotifShape, TemporalAttractorLibrary

MIN_SEQ_LEN = 3
MIN_RECURRENCE = 5
MIN_SIMILARITY = 0.75


class MotifDetector:
    """
    Detects motif candidates from a rolling window of pulse VSA vectors.
    Candidate: seq >= 3 pulses, recurrence >= 5, similarity >= 0.75
    """

    def __init__(self, dim: int, tal: TemporalAttractorLibrary, window: int = 20):
        self.dim = dim
        self.tal = tal
        self.window = window
        self.pulse_history: List[torch.Tensor] = []
        self.candidate_counts: dict = {}  # hash -> count

    def record_pulse(self, pulse_vsa: torch.Tensor):
        self.pulse_history.append(pulse_vsa.detach())
        if len(self.pulse_history) > self.window:
            self.pulse_history.pop(0)

    def _seq_hash(self, seq: List[torch.Tensor]) -> str:
        """Approximate hash via argmax of bundled sequence."""
        bundled = normalize(bundle(seq))
        return str(int(bundled.argmax().item()))

    def detect(self, trigger_vsa: torch.Tensor,
               outcome_vsa: torch.Tensor) -> Optional[TemporalMotifShape]:
        """
        Check if recent pulse history forms a motif candidate.
        Returns a TemporalMotifShape if candidate qualifies, else None.
        """
        if len(self.pulse_history) < MIN_SEQ_LEN:
            return None

        # Try all subsequences of length >= MIN_SEQ_LEN
        for seq_len in range(MIN_SEQ_LEN, len(self.pulse_history) + 1):
            seq = self.pulse_history[-seq_len:]
            seq_hash = self._seq_hash(seq)
            self.candidate_counts[seq_hash] = self.candidate_counts.get(seq_hash, 0) + 1

            if self.candidate_counts[seq_hash] >= MIN_RECURRENCE:
                # Check similarity against existing TAL entries
                bundled_seq = normalize(bundle(seq))
                existing = self.tal.retrieve(trigger_vsa, k=1)
                if existing:
                    sim, _ = existing[0]
                    if sim >= MIN_SIMILARITY:
                        # Already stored; update feedback
                        continue

                # Build ordering encoding (positional sum)
                pos_vecs = [v * (i + 1) / seq_len for i, v in enumerate(seq)]
                ordering_enc = normalize(bundle(pos_vecs))

                motif = TemporalMotifShape(
                    pulse_sequence_vsa=bundled_seq,
                    ordering_encoding=ordering_enc,
                    duration=seq_len,
                    trigger_vsa=trigger_vsa.detach(),
                    outcome_vsa=outcome_vsa.detach(),
                    success_rate=0.5,
                    pulse_savings=float(seq_len - 1),
                    recurrence=self.candidate_counts[seq_hash],
                )
                self.tal.store(motif)
                return motif

        return None
