"""USL Syntax Decoder — reconstruct token sequence from bond structure."""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple

from psn2.usl.linguistic_bonds import LinguisticBondTypes


class SyntaxDecoder(nn.Module):
    """Reconstructs an ordered token sequence from a committed shape + bond structure."""

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.token_proj = nn.Linear(dim, vocab_size)
        self.order_proj = nn.Linear(dim, 1)  # scalar position score

    def forward(self, shape_vec: torch.Tensor, n_tokens: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        shape_vec: [dim]
        Returns (token_logits [n_tokens, vocab_size], position_scores [n_tokens])
        """
        # Expand shape into n_tokens slots via learned perturbations
        slots = shape_vec.unsqueeze(0).expand(n_tokens, -1)
        token_logits = self.token_proj(slots)
        position_scores = self.order_proj(slots).squeeze(-1)
        return token_logits, position_scores
