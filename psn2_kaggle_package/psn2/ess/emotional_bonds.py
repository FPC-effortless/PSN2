"""T16: Emotional bonds — bind(perm_EMO(E.semantic_v), S.centroid_nu)."""
from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import List

from psn2.vsa import bind
from psn2.bonds import PermutationIndex

# Emotional bond permutation index base (offset from regular bond types)
EMO_PERM_BASE = 100


@dataclass
class EmotionalBond:
    emo_type: str
    emo_shape_idx: int
    target_shape_id: int
    bond_vector: torch.Tensor
    strength: float = 1.0

    def decay(self, lambda_emo: float = 0.85):
        self.strength *= lambda_emo


class EmotionalBondManager:
    """Creates and manages emotional bonds between emotional shapes and cognitive shapes."""

    def __init__(self, dim: int):
        self.dim = dim
        self.bonds: List[EmotionalBond] = []

    def form_bond(self, emo_type: str, emo_idx: int, emo_semantic_v: torch.Tensor,
                  target_shape_id: int, target_centroid: torch.Tensor) -> EmotionalBond:
        """bind(perm_EMO(E.semantic_v), S.centroid_nu)"""
        # Use EMO_PERM_BASE + emo type index as permutation seed
        from psn2.ess.emotional_shapes import EMOTIONAL_TYPES
        type_idx = EMOTIONAL_TYPES.index(emo_type) if emo_type in EMOTIONAL_TYPES else 0
        perm_id = EMO_PERM_BASE + type_idx
        permuted = PermutationIndex.apply(emo_semantic_v, perm_id)
        bond_vec = bind(permuted, target_centroid)
        bond = EmotionalBond(
            emo_type=emo_type,
            emo_shape_idx=emo_idx,
            target_shape_id=target_shape_id,
            bond_vector=bond_vec.detach(),
            strength=1.0,
        )
        self.bonds.append(bond)
        return bond

    def pulse_decay(self):
        self.bonds = [b for b in self.bonds if b.strength > 0.01]
        for b in self.bonds:
            b.decay()

    def state_dict(self) -> list:
        return [
            {
                "emo_type": b.emo_type,
                "emo_shape_idx": b.emo_shape_idx,
                "target_shape_id": b.target_shape_id,
                "bond_vector": b.bond_vector.tolist(),
                "strength": b.strength,
            }
            for b in self.bonds
        ]

    def load_state_dict(self, state: list):
        self.bonds = []
        for s in state:
            b = EmotionalBond(
                emo_type=s["emo_type"],
                emo_shape_idx=s["emo_shape_idx"],
                target_shape_id=s["target_shape_id"],
                bond_vector=torch.tensor(s["bond_vector"]),
                strength=s["strength"],
            )
            self.bonds.append(b)
