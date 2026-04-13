"""T24: Temporal Attractor Library (TAL) — store, retrieve, promote, retire temporal motifs."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

TAU_MOTIF = 0.65
TAU_MOTIF_SUCCESS = 0.75
TAU_MOTIF_SAVINGS = 3
TAU_MOTIF_RECURRENCE = 10
TAU_MOTIF_INTERFERENCE = 0.03


@dataclass
class TemporalMotifShape:
    pulse_sequence_vsa: torch.Tensor   # bundled VSA of pulse sequence
    ordering_encoding: torch.Tensor    # positional encoding of pulse order
    duration: int                      # number of pulses
    trigger_vsa: torch.Tensor
    outcome_vsa: torch.Tensor
    success_rate: float = 0.0
    pulse_savings: float = 0.0
    recurrence: int = 0
    promoted: bool = False
    nodes_freed: int = 0               # nodes whose spawn_pressure cleared by this motif

    def meets_promotion_gates(self) -> bool:
        return (
            self.recurrence >= TAU_MOTIF_RECURRENCE
            and self.success_rate >= TAU_MOTIF_SUCCESS
            and self.pulse_savings >= TAU_MOTIF_SAVINGS
        )


class TemporalAttractorLibrary:
    """Stores and retrieves temporal motif shapes."""

    def __init__(self, dim: int, max_size: int = 1000):
        self.dim = dim
        self.max_size = max_size
        self.motifs: List[TemporalMotifShape] = []

    def store(self, motif: TemporalMotifShape) -> int:
        if len(self.motifs) >= self.max_size:
            self._retire_lowest()
        self.motifs.append(motif)
        return len(self.motifs) - 1

    def _retire_lowest(self):
        if not self.motifs:
            return
        worst = min(range(len(self.motifs)),
                    key=lambda i: self.motifs[i].success_rate * self.motifs[i].recurrence)
        self.motifs.pop(worst)

    def retrieve(self, trigger_vsa: torch.Tensor, k: int = 3) -> List[Tuple[float, TemporalMotifShape]]:
        if not self.motifs:
            return []
        codebook = torch.stack([m.trigger_vsa for m in self.motifs], dim=0).to(trigger_vsa.device)
        sims = F.cosine_similarity(trigger_vsa.unsqueeze(0), codebook, dim=-1)
        k = min(k, len(self.motifs))
        vals, idxs = torch.topk(sims, k)
        results = []
        for i in range(k):
            sim = float(vals[i].item())
            if sim >= TAU_MOTIF:
                results.append((sim, self.motifs[int(idxs[i].item())]))
        return results

    def promote_eligible(self) -> List[TemporalMotifShape]:
        promoted = []
        for m in self.motifs:
            if not m.promoted and m.meets_promotion_gates():
                m.promoted = True
                promoted.append(m)
        return promoted

    def update_feedback(self, motif_idx: int, success: bool, savings: float):
        if 0 <= motif_idx < len(self.motifs):
            m = self.motifs[motif_idx]
            m.recurrence += 1
            alpha = 0.1
            m.success_rate = (1 - alpha) * m.success_rate + alpha * (1.0 if success else 0.0)
            m.pulse_savings = (1 - alpha) * m.pulse_savings + alpha * savings

    def state_dict(self) -> list:
        return [
            {
                "pulse_sequence_vsa": m.pulse_sequence_vsa.tolist(),
                "ordering_encoding": m.ordering_encoding.tolist(),
                "duration": m.duration,
                "trigger_vsa": m.trigger_vsa.tolist(),
                "outcome_vsa": m.outcome_vsa.tolist(),
                "success_rate": m.success_rate,
                "pulse_savings": m.pulse_savings,
                "recurrence": m.recurrence,
                "promoted": m.promoted,
                "nodes_freed": m.nodes_freed,
            }
            for m in self.motifs
        ]

    def load_state_dict(self, state: list):
        self.motifs = []
        for s in state:
            m = TemporalMotifShape(
                pulse_sequence_vsa=torch.tensor(s["pulse_sequence_vsa"]),
                ordering_encoding=torch.tensor(s["ordering_encoding"]),
                duration=s["duration"],
                trigger_vsa=torch.tensor(s["trigger_vsa"]),
                outcome_vsa=torch.tensor(s["outcome_vsa"]),
                success_rate=s["success_rate"],
                pulse_savings=s["pulse_savings"],
                recurrence=s["recurrence"],
                promoted=s["promoted"],
                nodes_freed=s.get("nodes_freed", 0),
            )
            self.motifs.append(m)
