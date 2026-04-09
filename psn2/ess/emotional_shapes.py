"""T16: Emotional Shape System — 8 emotional types with bond-mediated influence."""
from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional

EMOTIONAL_TYPES = [
    "threat_fear",
    "uncertainty_discomfort",
    "curiosity_drive",
    "confidence",
    "urgency",
    "satisfaction",
    "frustration",
    "aesthetic_resonance",
]

# Signed effect on bonded shape commit threshold (delta_theta_commit)
# PRD Section 7.1: threat_fear raises by +0.20*strength; others as specified
DELTA_THETA_COMMIT: Dict[str, float] = {
    "threat_fear":            +0.20,   # raises threshold (harder to commit)
    "uncertainty_discomfort": +0.10,
    "curiosity_drive":        -0.10,   # lowers threshold (easier to commit)
    "confidence":             -0.15,   # PRD: lowers render theta_commit by -0.15
    "urgency":                +0.10,   # PRD: raises tau_dissolve+0.10
    "satisfaction":           +0.30,   # PRD: increases utility_score multiplier by +0.30
    "frustration":            +0.08,
    "aesthetic_resonance":    -0.06,
}

LAMBDA_EMO = 0.85   # faster decay than cognitive bonds
TAU_EMO_STABLE = 3  # pulses before emotional shape is considered stable


@dataclass
class EmotionalShape:
    emo_type: str
    semantic_v: torch.Tensor
    strength: float = 1.0
    age: int = 0
    bonded_shape_ids: List[int] = field(default_factory=list)

    def decay(self):
        self.strength *= LAMBDA_EMO
        self.age += 1

    @property
    def is_stable(self) -> bool:
        return self.age >= TAU_EMO_STABLE and self.strength > 0.1

    @property
    def delta_theta(self) -> float:
        return DELTA_THETA_COMMIT.get(self.emo_type, 0.0) * self.strength


class EmotionalShapeInducer(nn.Module):
    """Morphogenic MLP that induces emotional shapes in Phase C."""

    def __init__(self, dim: int, n_types: int = len(EMOTIONAL_TYPES)):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, n_types),
        )
        self.semantic_proj = nn.Linear(dim, dim)

    def forward(self, shape_centroid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (type_logits [n_types], semantic_v [dim])."""
        logits = self.classifier(shape_centroid)
        semantic_v = torch.tanh(self.semantic_proj(shape_centroid))
        return logits, semantic_v


class EmotionalShapeSystem:
    """Manages the lifecycle of emotional shapes."""

    def __init__(self, dim: int):
        self.dim = dim
        self.inducer = EmotionalShapeInducer(dim)
        self.active: List[EmotionalShape] = []

    def induce(self, shape_centroid: torch.Tensor, threshold: float = 0.5) -> Optional[EmotionalShape]:
        """Phase C: attempt to induce an emotional shape from a shape centroid."""
        with torch.no_grad():
            logits, semantic_v = self.inducer(shape_centroid)
            probs = torch.softmax(logits, dim=-1)
            max_prob, idx = probs.max(dim=-1)
            if max_prob.item() < threshold:
                return None
            emo_type = EMOTIONAL_TYPES[int(idx.item())]
        emo = EmotionalShape(emo_type=emo_type, semantic_v=semantic_v.detach(), strength=float(max_prob.item()))
        self.active.append(emo)
        return emo

    def pulse_decay(self):
        """Decay all emotional shapes; remove dead ones."""
        self.active = [e for e in self.active if e.strength > 0.01]
        for e in self.active:
            e.decay()

    def get_threshold_delta(self, shape_id: int) -> float:
        """Aggregate delta_theta_commit for a given bonded shape."""
        delta = 0.0
        for e in self.active:
            if shape_id in e.bonded_shape_ids:
                delta += e.delta_theta
        return delta

    def state_dict(self) -> list:
        return [
            {
                "emo_type": e.emo_type,
                "semantic_v": e.semantic_v.tolist(),
                "strength": e.strength,
                "age": e.age,
                "bonded_shape_ids": e.bonded_shape_ids,
            }
            for e in self.active
        ]

    def load_state_dict(self, state: list):
        self.active = []
        for s in state:
            e = EmotionalShape(
                emo_type=s["emo_type"],
                semantic_v=torch.tensor(s["semantic_v"]),
                strength=s["strength"],
                age=s["age"],
                bonded_shape_ids=s["bonded_shape_ids"],
            )
            self.active.append(e)
