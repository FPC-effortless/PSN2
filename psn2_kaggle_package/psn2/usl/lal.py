"""T18: Lexical Attractor Library (LAL) — word shape storage and retrieval.

Dimensional note (addresses word shape mismatch critique):
  A WordShape has total dimensionality 512+256+128+16 = 912 dims across
  four geometry spaces. The substrate VSA space is D=512. These are NOT
  concatenated — each geometry lives in its own space and is projected
  into the substrate D-dim space only when needed for VSA operations.
  The semantic_v IS the substrate-space vector (R^D). The syntactic_v,
  pragmatic_v, and freq_v are auxiliary geometry spaces used by the
  syntax encoder and pragmatics renderer respectively. They never enter
  the main VSA algebra directly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

D_SYNTACTIC = 256
D_PRAGMATIC = 128
D_FREQ = 16


@dataclass
class WordShape:
    token: str
    semantic_v: torch.Tensor    # R^D (vsa_dim)
    syntactic_v: torch.Tensor   # R^512
    pragmatic_v: torch.Tensor   # R^256
    freq_v: torch.Tensor        # R^16
    phi_i: float = 0.0          # integration complexity
    frequency: int = 1


class LexicalAttractorLibrary:
    """
    Stores up to LAL_vocab_size word shapes.
    Kaggle start: 5k entries.
    Retrieval via VSA cleanup projection (cosine top-k).
    """

    def __init__(self, dim: int, max_size: int = 8192):
        self.dim = dim
        self.max_size = max_size
        self.shapes: List[WordShape] = []
        self._token_index: Dict[str, int] = {}

    def add(self, word_shape: WordShape) -> int:
        """Add a word shape; return its index."""
        if word_shape.token in self._token_index:
            # Merge: update frequency
            idx = self._token_index[word_shape.token]
            self.shapes[idx].frequency += 1
            return idx
        if len(self.shapes) >= self.max_size:
            self._prune()
        idx = len(self.shapes)
        self.shapes.append(word_shape)
        self._token_index[word_shape.token] = idx
        return idx

    def _prune(self, keep_fraction: float = 0.8):
        """Prune lowest-frequency entries."""
        keep = int(self.max_size * keep_fraction)
        sorted_shapes = sorted(self.shapes, key=lambda s: s.frequency, reverse=True)
        self.shapes = sorted_shapes[:keep]
        self._token_index = {s.token: i for i, s in enumerate(self.shapes)}

    def retrieve_by_token(self, token: str) -> Optional[WordShape]:
        idx = self._token_index.get(token)
        return self.shapes[idx] if idx is not None else None

    def retrieve_by_vector(self, query: torch.Tensor, k: int = 5) -> List[Tuple[float, WordShape]]:
        """Cosine top-k retrieval against semantic vectors."""
        if not self.shapes:
            return []
        codebook = torch.stack([s.semantic_v for s in self.shapes], dim=0).to(query.device)
        sims = F.cosine_similarity(query.unsqueeze(0), codebook, dim=-1)
        k = min(k, len(self.shapes))
        vals, idxs = torch.topk(sims, k)
        return [(float(vals[i].item()), self.shapes[int(idxs[i].item())]) for i in range(k)]

    def make_word_shape(self, token: str, semantic_v: torch.Tensor) -> WordShape:
        """Create a word shape with random syntactic/pragmatic/freq vectors."""
        device = semantic_v.device
        return WordShape(
            token=token,
            semantic_v=semantic_v,
            syntactic_v=F.normalize(torch.randn(D_SYNTACTIC, device=device), dim=-1),
            pragmatic_v=F.normalize(torch.randn(D_PRAGMATIC, device=device), dim=-1),
            freq_v=torch.randn(D_FREQ, device=device),
            phi_i=0.0,
            frequency=1,
        )

    def __len__(self) -> int:
        return len(self.shapes)

    def state_dict(self) -> list:
        return [
            {
                "token": s.token,
                "semantic_v": s.semantic_v.tolist(),
                "syntactic_v": s.syntactic_v.tolist(),
                "pragmatic_v": s.pragmatic_v.tolist(),
                "freq_v": s.freq_v.tolist(),
                "phi_i": s.phi_i,
                "frequency": s.frequency,
            }
            for s in self.shapes
        ]

    def load_state_dict(self, state: list):
        self.shapes = []
        self._token_index = {}
        for i, s in enumerate(state):
            ws = WordShape(
                token=s["token"],
                semantic_v=torch.tensor(s["semantic_v"]),
                syntactic_v=torch.tensor(s["syntactic_v"]),
                pragmatic_v=torch.tensor(s["pragmatic_v"]),
                freq_v=torch.tensor(s["freq_v"]),
                phi_i=s["phi_i"],
                frequency=s["frequency"],
            )
            self.shapes.append(ws)
            self._token_index[ws.token] = i
