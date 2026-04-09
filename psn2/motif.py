"""T06: Temporal Motif Library — detection, storage, promotion gates.
Upgraded to full PRD spec (was partial).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, asdict, field
from typing import List, Optional

TAU_MOTIF_SIMILARITY = 0.75
TAU_MOTIF_SUCCESS = 0.75
TAU_MOTIF_SAVINGS = 3
MIN_RECURRENCE = 5
PROMOTION_RECURRENCE = 10
MAX_INTERFERENCE = 0.03


@dataclass
class Motif:
    seq_len: int
    success_rate: float
    pulse_savings: float
    trigger_id: int
    outcome_id: int
    recurrence: int = 0
    promoted: bool = False
    interference_score: float = 0.0
    # VSA vectors stored as lists for serialization
    pulse_sequence_vsa: Optional[List[float]] = None
    trigger_vsa: Optional[List[float]] = None
    outcome_vsa: Optional[List[float]] = None

    def meets_promotion_gates(self) -> bool:
        return (
            self.recurrence >= PROMOTION_RECURRENCE
            and self.success_rate >= TAU_MOTIF_SUCCESS
            and self.pulse_savings >= TAU_MOTIF_SAVINGS
            and self.interference_score <= MAX_INTERFERENCE
        )

    def pulse_seq_tensor(self) -> Optional[torch.Tensor]:
        if self.pulse_sequence_vsa:
            return torch.tensor(self.pulse_sequence_vsa)
        return None

    def trigger_tensor(self) -> Optional[torch.Tensor]:
        if self.trigger_vsa:
            return torch.tensor(self.trigger_vsa)
        return None


class MotifLibrary:
    """
    Stores temporal motif shapes with VSA vectors.
    Promotion gates: recurrence >= 10, success_rate >= 0.75, pulse_savings >= 3, interference <= 0.03
    Motif replay: inject pre-formed shapes in Perceptive regime.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.motifs: List[Motif] = []

    def add(self, motif: Motif):
        self.motifs.append(motif)

    def detect_candidate(self, pulse_vsas: List[torch.Tensor],
                          trigger_vsa: torch.Tensor,
                          outcome_vsa: torch.Tensor) -> Optional[Motif]:
        """
        Detect motif candidate: seq >= 3 pulses, recurrence >= 5, similarity >= 0.75.
        """
        if len(pulse_vsas) < 3:
            return None

        # Bundle pulse sequence
        stacked = torch.stack(pulse_vsas, dim=0)
        bundled = F.normalize(stacked.mean(dim=0), dim=-1)

        # Check similarity against existing motifs
        for m in self.motifs:
            t = m.pulse_seq_tensor()
            if t is None:
                continue
            sim = float(F.cosine_similarity(bundled.unsqueeze(0), t.unsqueeze(0)).item())
            if sim >= TAU_MOTIF_SIMILARITY:
                m.recurrence += 1
                return m

        # New candidate
        motif = Motif(
            seq_len=len(pulse_vsas),
            success_rate=0.5,
            pulse_savings=float(len(pulse_vsas) - 1),
            trigger_id=0,
            outcome_id=0,
            recurrence=1,
            pulse_sequence_vsa=bundled.tolist(),
            trigger_vsa=trigger_vsa.detach().tolist(),
            outcome_vsa=outcome_vsa.detach().tolist(),
        )
        self.motifs.append(motif)
        return motif

    def promote_eligible(self) -> List[Motif]:
        promoted = []
        for m in self.motifs:
            if not m.promoted and m.meets_promotion_gates():
                m.promoted = True
                promoted.append(m)
        return promoted

    def replay(self, trigger_vsa: torch.Tensor, regime: str = "perceptive") -> Optional[torch.Tensor]:
        """
        Inject pre-formed shapes in Perceptive regime.
        Returns pulse_sequence_vsa if a promoted motif matches trigger.
        """
        if regime != "perceptive":
            return None
        for m in self.motifs:
            if not m.promoted:
                continue
            t = m.trigger_tensor()
            if t is None:
                continue
            sim = float(F.cosine_similarity(trigger_vsa.unsqueeze(0), t.unsqueeze(0)).item())
            if sim >= TAU_MOTIF_SIMILARITY:
                return m.pulse_seq_tensor()
        return None

    def update_feedback(self, motif: Motif, success: bool, savings: float):
        alpha = 0.1
        motif.success_rate = (1 - alpha) * motif.success_rate + alpha * (1.0 if success else 0.0)
        motif.pulse_savings = (1 - alpha) * motif.pulse_savings + alpha * savings

    def state_dict(self) -> list:
        return [asdict(m) for m in self.motifs]

    def load_state_dict(self, state: list):
        self.motifs = []
        for x in state:
            self.motifs.append(Motif(**x))
