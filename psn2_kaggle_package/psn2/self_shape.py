"""T26: Self Shape — persistent model of system state, competence, goals, identity."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from psn2.vsa import normalize, bundle


@dataclass
class SelfShape:
    """
    Persistent self-model. Bonds to emotional shapes and goal shapes.
    Regulatory bonds raise suppression probability on identity-violating shapes.
    Attention-narrowing bond to active task shape.
    Aesthetic self-shape with harmonic geometry preferences.
    """
    dim: int
    semantic_v: torch.Tensor
    competence: Dict[str, float] = field(default_factory=dict)
    identity_constraints: List[torch.Tensor] = field(default_factory=list)
    active_task_vsa: Optional[torch.Tensor] = None
    aesthetic_v: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.aesthetic_v is None:
            # Harmonic geometry: initialize as normalized random
            self.aesthetic_v = normalize(torch.randn(self.dim))

    def update(self, new_state_vsa: torch.Tensor, alpha: float = 0.05):
        """Continuous update during operation."""
        self.semantic_v = normalize((1 - alpha) * self.semantic_v + alpha * new_state_vsa)

    def set_active_task(self, task_vsa: torch.Tensor):
        """Attention-narrowing bond to active task shape."""
        self.active_task_vsa = task_vsa.detach()

    def suppression_probability(self, shape_vsa: torch.Tensor) -> float:
        """
        Regulatory bonds: identity constraints raise suppression probability
        on shapes that violate identity.
        """
        if not self.identity_constraints:
            return 0.0
        violations = []
        for constraint in self.identity_constraints:
            sim = float(F.cosine_similarity(shape_vsa.unsqueeze(0), constraint.unsqueeze(0)).item())
            # Low similarity to identity constraint = potential violation
            violations.append(max(0.0, 0.5 - sim))
        return min(1.0, sum(violations) / len(violations))

    def add_identity_constraint(self, constraint_vsa: torch.Tensor):
        self.identity_constraints.append(constraint_vsa.detach())

    def update_competence(self, domain: str, score: float, alpha: float = 0.1):
        prev = self.competence.get(domain, 0.5)
        self.competence[domain] = (1 - alpha) * prev + alpha * score

    def state_dict(self) -> dict:
        return {
            "semantic_v": self.semantic_v.tolist(),
            "competence": self.competence,
            "identity_constraints": [c.tolist() for c in self.identity_constraints],
            "active_task_vsa": self.active_task_vsa.tolist() if self.active_task_vsa is not None else None,
            "aesthetic_v": self.aesthetic_v.tolist() if self.aesthetic_v is not None else None,
        }

    @classmethod
    def from_state_dict(cls, s: dict, dim: int) -> "SelfShape":
        obj = cls(
            dim=dim,
            semantic_v=torch.tensor(s["semantic_v"]),
            competence=s["competence"],
            identity_constraints=[torch.tensor(c) for c in s["identity_constraints"]],
            active_task_vsa=torch.tensor(s["active_task_vsa"]) if s["active_task_vsa"] else None,
            aesthetic_v=torch.tensor(s["aesthetic_v"]) if s["aesthetic_v"] else None,
        )
        return obj
