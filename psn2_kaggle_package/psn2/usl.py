from __future__ import annotations

import torch
import torch.nn as nn

class USLEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, tokens: torch.Tensor):
        return self.emb(tokens)

class USLDecoder(nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size)

    def forward(self, shapes: torch.Tensor):
        return self.proj(shapes)
