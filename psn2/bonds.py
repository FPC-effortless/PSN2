"""T15: Bond System — typed VSA bonds with permutation indices and legality matrix."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .vsa import bind, cosine, cleanup

# Bond type indices (0-3 cognitive, 10-19 linguistic)
BOND_TYPES = {
    "causal":     0,
    "temporal":   1,
    "spatial":    2,
    "part_whole": 3,
    # Linguistic bonds P_10–P_19
    "MODIFIES":            10,
    "DETERMINES":          11,
    "SUBJECT_OF":          12,
    "OBJECT_OF":           13,
    "PREDICATE_OF":        14,
    "DISCOURSE_CONTINUES": 15,
    "DISCOURSE_CONTRASTS": 16,
    "DISCOURSE_GROUNDS":   17,
    "COREFERS_WITH":       18,
    "SCOPES_OVER":         19,
}

BOND_TYPE_NAMES = {v: k for k, v in BOND_TYPES.items()}

# Legality matrix: shape_type -> set of allowed bond type indices
LEGALITY_MATRIX: Dict[str, List[int]] = {
    "perceptive":    [0, 1, 2, 3],
    "compositional": [0, 1, 2, 3, 10, 11, 12, 13, 14],
    "recursive":     [0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "semantic":      [0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "goal":          [0, 1],
    "emotional":     [0, 1, 2, 3],
    "social":        [0, 1, 2, 3, 15, 16, 17, 18],
}


class PermutationIndex:
    """Deterministic permutation per bond type, seeded from bond type id."""
    _cache: Dict[Tuple[int, int], torch.Tensor] = {}

    @classmethod
    def get(cls, bond_type_id: int, dim: int, device=None) -> torch.Tensor:
        key = (bond_type_id, dim)
        if key not in cls._cache:
            g = torch.Generator()
            g.manual_seed(bond_type_id + 1337)
            perm = torch.randperm(dim, generator=g)
            cls._cache[key] = perm
        return cls._cache[key].to(device)

    @classmethod
    def apply(cls, vec: torch.Tensor, bond_type_id: int) -> torch.Tensor:
        perm = cls.get(bond_type_id, vec.shape[-1], vec.device)
        return vec[..., perm]

    @classmethod
    def invert(cls, vec: torch.Tensor, bond_type_id: int) -> torch.Tensor:
        perm = cls.get(bond_type_id, vec.shape[-1], vec.device)
        inv_perm = torch.argsort(perm)
        return vec[..., inv_perm]


@dataclass
class Bond:
    bond_type: str
    source_id: int
    target_id: int
    bond_vector: torch.Tensor   # bind(perm_TYPE(a.semantic_v), b.semantic_v)
    strength: float = 1.0
    age: int = 0

    def decay(self, lambda_bond: float = 0.90):
        self.strength *= lambda_bond
        self.age += 1


class BondSystem:
    """Manages typed VSA bonds between nodes."""

    LAMBDA_BOND = 0.90

    def __init__(self, dim: int):
        self.dim = dim
        self.bonds: List[Bond] = []

    def form_bond(self, bond_type: str, src_id: int, tgt_id: int,
                  src_vec: torch.Tensor, tgt_vec: torch.Tensor,
                  shape_type: str = "compositional") -> Optional[Bond]:
        """Form a bond if legal for the given shape type.

        Fix #3: normalize vectors before bind so unbind math is correct
        regardless of whether nu is bipolar or continuous. For normalized
        vectors, bind(a,b) = a*b and unbind via bond*b recovers a up to
        the normalization factor, which cleanup handles via cosine similarity.
        """
        type_id = BOND_TYPES.get(bond_type)
        if type_id is None:
            raise ValueError(f"Unknown bond type: {bond_type}")
        allowed = LEGALITY_MATRIX.get(shape_type, [])
        if type_id not in allowed:
            return None
        # Normalize before bind so unbind is well-defined for non-bipolar vectors
        src_norm = F.normalize(src_vec.flatten(), dim=0)
        tgt_norm = F.normalize(tgt_vec.flatten(), dim=0)
        permuted = PermutationIndex.apply(src_norm, type_id)
        bond_vec = bind(permuted, tgt_norm)
        bond = Bond(bond_type=bond_type, source_id=src_id, target_id=tgt_id,
                    bond_vector=bond_vec.detach(), strength=1.0)
        self.bonds.append(bond)
        return bond

    def recover_source(self, bond: Bond, tgt_vec: torch.Tensor,
                       codebook: torch.Tensor) -> Tuple[int, torch.Tensor, float]:
        """Unbind + cleanup to recover source vector.

        Fix #3: normalize tgt_vec before unbind to match the normalized
        tgt_norm used during form_bond.
        """
        type_id = BOND_TYPES[bond.bond_type]
        tgt_norm = F.normalize(tgt_vec.flatten(), dim=0)
        # unbind: bond_vec * tgt_norm recovers perm(src_norm) for normalized vectors
        unbound = bond.bond_vector * tgt_norm
        recovered = PermutationIndex.invert(unbound, type_id)
        idx, vec, sim = cleanup(recovered, codebook)
        return int(idx), vec, float(sim)

    def pulse_decay(self):
        """Decay all bond strengths; remove dead bonds."""
        self.bonds = [b for b in self.bonds if b.strength > 0.01]
        for b in self.bonds:
            b.decay(self.LAMBDA_BOND)

    def get_bonds_for(self, node_id: int) -> List[Bond]:
        return [b for b in self.bonds if b.source_id == node_id or b.target_id == node_id]

    def state_dict(self) -> list:
        out = []
        for b in self.bonds:
            out.append({
                "bond_type": b.bond_type,
                "source_id": b.source_id,
                "target_id": b.target_id,
                "bond_vector": b.bond_vector.tolist(),
                "strength": b.strength,
                "age": b.age,
            })
        return out

    def load_state_dict(self, state: list):
        self.bonds = []
        for s in state:
            b = Bond(
                bond_type=s["bond_type"],
                source_id=s["source_id"],
                target_id=s["target_id"],
                bond_vector=torch.tensor(s["bond_vector"]),
                strength=s["strength"],
                age=s["age"],
            )
            self.bonds.append(b)
