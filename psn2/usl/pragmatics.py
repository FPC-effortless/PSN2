"""USL Pragmatics Renderer — surface generation from committed state."""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import List


class PragmaticsRenderer(nn.Module):
    """Renders a surface string from committed shape + decoded tokens."""

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        # Context-aware re-ranking of token logits
        self.context_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, vocab_size),
        )

    def forward(self, shape_vec: torch.Tensor, token_logits: torch.Tensor) -> torch.Tensor:
        """
        shape_vec: [dim]
        token_logits: [n_tokens, vocab_size]
        Returns refined token_logits [n_tokens, vocab_size].
        """
        n_tokens = token_logits.size(0)
        ctx = shape_vec.unsqueeze(0).expand(n_tokens, -1)
        # Combine shape context with token logits via a gate
        combined = torch.cat([ctx, token_logits[..., :self.dim].clamp(-1, 1)], dim=-1)
        gate = torch.sigmoid(self.context_gate(combined))
        return token_logits + gate * 0.1
