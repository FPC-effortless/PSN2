"""T19: Linguistic Bond Types — 10 bond types with VSA permutation indices P_10–P_19."""
from __future__ import annotations

import torch
from typing import Dict, List, Tuple

from psn2.vsa import bind, cleanup
from psn2.bonds import PermutationIndex, LEGALITY_MATRIX

# P_10 through P_19
LINGUISTIC_BOND_TYPES: Dict[str, int] = {
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

LINGUISTIC_BOND_NAMES: Dict[int, str] = {v: k for k, v in LINGUISTIC_BOND_TYPES.items()}


class LinguisticBondTypes:
    """Provides bind/unbind/recover operations for all 10 linguistic bond types."""

    def __init__(self, dim: int):
        self.dim = dim

    def bind(self, bond_type: str, src_vec: torch.Tensor, tgt_vec: torch.Tensor) -> torch.Tensor:
        """bind(perm_TYPE(src_vec), tgt_vec)"""
        type_id = LINGUISTIC_BOND_TYPES[bond_type]
        permuted = PermutationIndex.apply(src_vec, type_id)
        return bind(permuted, tgt_vec)

    def unbind(self, bond_type: str, bond_vec: torch.Tensor, tgt_vec: torch.Tensor) -> torch.Tensor:
        """Recover src_vec: invert_perm(bond_vec * tgt_vec)"""
        type_id = LINGUISTIC_BOND_TYPES[bond_type]
        unbound = bond_vec * tgt_vec
        return PermutationIndex.invert(unbound, type_id)

    def recover(self, bond_type: str, bond_vec: torch.Tensor, tgt_vec: torch.Tensor,
                codebook: torch.Tensor) -> Tuple[int, torch.Tensor, float]:
        """Unbind + cleanup to recover source."""
        recovered = self.unbind(bond_type, bond_vec, tgt_vec)
        idx, vec, sim = cleanup(recovered, codebook)
        return int(idx), vec, float(sim)

    def legality_check(self, bond_type: str, shape_type: str) -> bool:
        type_id = LINGUISTIC_BOND_TYPES.get(bond_type)
        if type_id is None:
            return False
        return type_id in LEGALITY_MATRIX.get(shape_type, [])

    def all_types(self) -> List[str]:
        return list(LINGUISTIC_BOND_TYPES.keys())
