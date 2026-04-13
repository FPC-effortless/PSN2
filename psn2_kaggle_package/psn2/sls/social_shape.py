"""T23: Social Shape — agent model with trust, competence, reasoning style."""
from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Dict

TAU_SOCIAL_TRUST_MIN = 0.05
TAU_SOCIAL_TRUST_SUSPICIOUS = 0.30


@dataclass
class SocialShape:
    agent_id: str
    goal_vsa: torch.Tensor
    reasoning_style: str = "unknown"
    competence: Dict[str, float] = field(default_factory=dict)
    trust: float = 0.5

    def update_trust(self, verifier_agreed: bool):
        if verifier_agreed:
            self.trust = min(1.0, self.trust + 0.05)
        else:
            self.trust = max(TAU_SOCIAL_TRUST_MIN, self.trust - 0.15)

    @property
    def is_suspicious(self) -> bool:
        return self.trust < TAU_SOCIAL_TRUST_SUSPICIOUS

    def state_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "goal_vsa": self.goal_vsa.tolist(),
            "reasoning_style": self.reasoning_style,
            "competence": self.competence,
            "trust": self.trust,
        }

    @classmethod
    def from_state_dict(cls, s: dict) -> "SocialShape":
        return cls(
            agent_id=s["agent_id"],
            goal_vsa=torch.tensor(s["goal_vsa"]),
            reasoning_style=s["reasoning_style"],
            competence=s["competence"],
            trust=s["trust"],
        )
