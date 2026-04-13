"""T04: Attractor Library — codebook store, cosine retrieval, pruning by utility.
PRD Section 12.1/12.3: Lite/Kaggle max attractors = 2048.
Insert if no match (cos < 0.45). Prune lowest-utility entries when cap exceeded.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

# PRD Section 12.1: spawn trigger is "no attractor match above cos=0.45"
COS_INSERT_THRESHOLD = 0.45


@dataclass
class AttractorEntry:
    """PRD Section 12.1 attractor schema."""
    prototype_nu: List[float]
    variance: float = 0.0
    tier: str = "working"
    reuse_stats: int = 0
    decay_state: float = 1.0


class AttractorLibrary:
    # PRD Section 12.3: Lite/Kaggle max = 2048
    def __init__(self, dim: int, max_size: int = 2048):
        self.dim = dim
        self.max_size = max_size
        self.codebook: List[torch.Tensor] = []
        self.utility: List[float] = []
        self.entries: List[AttractorEntry] = []
        # Fix #15: cached tensor to avoid O(N) stack on every insert/query
        self._cache: Optional[torch.Tensor] = None
        self._cache_dirty: bool = False

    def _invalidate_cache(self):
        self._cache_dirty = True

    def as_tensor(self, device=None) -> torch.Tensor:
        """Return codebook as tensor, using cache when clean."""
        if not self.codebook:
            return torch.empty(0, self.dim, device=device)
        if self._cache is None or self._cache_dirty or self._cache.shape[0] != len(self.codebook):
            self._cache = torch.stack(list(self.codebook), dim=0)
            self._cache_dirty = False
        return self._cache.to(device) if device is not None else self._cache

    def add(self, vec: torch.Tensor, utility: float = 0.5) -> bool:
        """Insert if no match (cos < 0.45). Returns True if inserted."""
        vec = vec.detach().flatten()
        if vec.numel() != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vec.numel()}")
        # Check for existing match
        if self.codebook:
            cb = self.as_tensor(device=vec.device)
            sims = F.cosine_similarity(vec.unsqueeze(0), cb, dim=-1)
            if sims.max().item() >= COS_INSERT_THRESHOLD:
                return False
        if len(self.codebook) >= self.max_size:
            self._prune()
        self.codebook.append(vec.cpu())
        self.utility.append(utility)
        self.entries.append(AttractorEntry(prototype_nu=vec.cpu().tolist()))
        self._invalidate_cache()
        return True

    def _prune(self, keep: int = None):
        """Prune lowest-utility entries."""
        if keep is None:
            keep = int(self.max_size * 0.9)
        if len(self.codebook) <= keep:
            return
        order = sorted(range(len(self.utility)), key=lambda i: self.utility[i], reverse=True)
        order = order[:keep]
        self.codebook = [self.codebook[i] for i in order]
        self.utility = [self.utility[i] for i in order]
        self.entries = [self.entries[i] for i in order]
        self._invalidate_cache()

    def query(self, vec: torch.Tensor, k: int = 5) -> List[Tuple[float, torch.Tensor]]:
        """Cosine top-k retrieval."""
        cb = self.as_tensor(device=vec.device)
        if cb.numel() == 0:
            return []
        sims = F.cosine_similarity(vec.unsqueeze(0), cb, dim=-1)
        k = min(k, len(self.codebook))
        vals, idx = torch.topk(sims, k=k)
        return [(float(v), cb[i]) for v, i in zip(vals, idx)]

    def update_utility(self, idx: int, utility: float):
        if 0 <= idx < len(self.utility):
            self.utility[idx] = utility

    def prune(self, utility_tensor: torch.Tensor = None, keep: int = 5000):
        """External prune call with optional utility tensor override."""
        if utility_tensor is not None:
            self.utility = utility_tensor.tolist()[:len(self.codebook)]
        self._prune(keep=keep)

    def state_dict(self) -> dict:
        return {
            "codebook": [v.tolist() for v in self.codebook],
            "utility": list(self.utility),
            "entries": [asdict(e) for e in self.entries],
            "dim": self.dim,
            "max_size": self.max_size,
        }

    def load_state_dict(self, state: dict):
        self.codebook = [torch.tensor(v) for v in state.get("codebook", [])]
        self.utility = list(state.get("utility", [0.5] * len(self.codebook)))
        raw_entries = state.get("entries", [])
        self.entries = [AttractorEntry(**e) for e in raw_entries]
        # Back-fill entries if missing
        while len(self.entries) < len(self.codebook):
            self.entries.append(AttractorEntry(prototype_nu=self.codebook[len(self.entries)].tolist()))
        # Fix: Invalidate cache after loading state
        self._invalidate_cache()

    def __len__(self) -> int:
        return len(self.codebook)
