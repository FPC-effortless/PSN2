"""T20: Syntax Encoder — lightweight dependency parser producing bond proposals."""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple

from psn2.usl.linguistic_bonds import LINGUISTIC_BOND_TYPES

BondProposal = Tuple[int, int, str, float]  # (i, j, bond_type, strength)


class SyntaxEncoder(nn.Module):
    """
    Input:  sequence of word shape semantic vectors [seq_len, dim]
    Output: list of (i, j, bond_type, strength) bond proposals

    Architecture: small attention-based scorer over all pairs.
    """

    def __init__(self, dim: int, n_bond_types: int = len(LINGUISTIC_BOND_TYPES)):
        super().__init__()
        self.dim = dim
        self.n_bond_types = n_bond_types
        self.bond_type_names = list(LINGUISTIC_BOND_TYPES.keys())

        # Pair scorer: concat(src, tgt) -> bond_type logits + existence logit
        self.pair_scorer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, n_bond_types + 1),  # +1 for "no bond"
        )

    def forward(self, word_vecs: torch.Tensor,
                threshold: float = 0.3) -> List[BondProposal]:
        """
        word_vecs: [seq_len, dim]
        Returns list of (i, j, bond_type_name, strength).
        """
        seq_len = word_vecs.size(0)
        proposals: List[BondProposal] = []

        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    continue
                pair = torch.cat([word_vecs[i], word_vecs[j]], dim=-1)
                logits = self.pair_scorer(pair)
                probs = torch.softmax(logits, dim=-1)
                # Last index = "no bond"
                bond_probs = probs[:-1]
                no_bond_prob = probs[-1]
                if no_bond_prob.item() > (1 - threshold):
                    continue
                best_type_idx = int(bond_probs.argmax().item())
                strength = float(bond_probs[best_type_idx].item())
                if strength > threshold:
                    bond_type = self.bond_type_names[best_type_idx]
                    proposals.append((i, j, bond_type, strength))

        return proposals
