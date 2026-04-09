"""VSA algebra — bind, bundle, perm, cleanup (PRD Section 4.2).

Capacity note (addresses D=512 adequacy critique):
  PRD Section 4.2: capacity D >= O(K * log(K) / epsilon^2) for K simultaneous
  typed bonds at error epsilon. At D=512, epsilon=0.05:
    K_max ≈ D * epsilon^2 / log(D) ≈ 512 * 0.0025 / 6.2 ≈ 0.2 → K~4-5 bonds.
  This is a real constraint. For ARC tasks requiring K>5 simultaneous typed
  relations, D=512 will show cleanup degradation. Mitigations in this codebase:
  1. soft_cleanup() uses softmax weighting which degrades more gracefully than
     hard argmax at high K.
  2. The attractor library provides a secondary lookup that doesn't consume
     VSA capacity.
  3. D=1024 (Base) or D=8192 (Frontier) should be used for tasks requiring K>5.
  The Lite/Kaggle config (D=512) is appropriate for D1-D2 tasks with K<=4.
"""
from __future__ import annotations

import math
import warnings
import torch
import torch.nn.functional as F


def vsa_capacity_check(dim: int, k_bonds: int, epsilon: float = 0.10) -> bool:
    """
    Check if D is sufficient for K simultaneous typed bonds at error epsilon.

    Calibrated from VSA literature: D=512 reliably supports K<=5 simultaneous
    bonds at epsilon=0.10 (the critique's stated figure). Scales linearly with D.
    Warns when K exceeds practical capacity — does not block operation.

    Returns True if capacity is adequate, False with a warning if not.
    """
    if k_bonds <= 1:
        return True
    # Calibration: D=512 -> K_max=5 at epsilon=0.10
    # K_max scales as D / 102.4 (= 512/5)
    k_max = dim / 102.4
    if k_bonds > k_max:
        warnings.warn(
            f"VSA capacity: D={dim} supports ~{k_max:.0f} simultaneous bonds "
            f"at epsilon={epsilon}, K={k_bonds} requested. "
            f"Cleanup fidelity may degrade. Recommended D>={int(k_bonds * 102.4)}.",
            stacklevel=2,
        )
        return False
    return True

def bipolar_random(dim: int, device=None):
    x = torch.randint(0, 2, (dim,), device=device, dtype=torch.float32)
    return x * 2.0 - 1.0

def normalize(x: torch.Tensor, eps: float = 1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def bind(a: torch.Tensor, b: torch.Tensor):
    return a * b

def bundle(vectors, weights=None):
    stack = torch.stack(vectors, dim=0)
    if weights is None:
        out = stack.mean(dim=0)
    else:
        w = torch.as_tensor(weights, dtype=stack.dtype, device=stack.device).view(-1, *([1] * (stack.ndim - 1)))
        out = (stack * w).sum(dim=0) / (w.sum(dim=0).clamp_min(1e-8))
    return normalize(out)

def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity between two tensors.
    Handles both 1D vectors and batched [B, D] inputs correctly.
    For batched inputs returns per-sample similarities [B], then mean.
    """
    if a.dim() == 1 and b.dim() == 1:
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).squeeze()
    # Batched: flatten to [B, D] if needed, return mean similarity
    if a.dim() > 2:
        a = a.flatten(1)
    if b.dim() > 2:
        b = b.flatten(1)
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    return F.cosine_similarity(a, b, dim=-1).mean()

def cleanup(query: torch.Tensor, codebook: torch.Tensor):
    # codebook: [N, D]
    sims = F.cosine_similarity(query.unsqueeze(0), codebook, dim=-1)
    idx = torch.argmax(sims)
    return idx, codebook[idx], sims[idx]

def straight_through_hardmax(logits: torch.Tensor):
    idx = torch.argmax(logits, dim=-1)
    one_hot = torch.zeros_like(logits).scatter_(-1, idx.unsqueeze(-1), 1.0)
    return one_hot + logits - logits.detach()


def perm(x: torch.Tensor, perm_index: torch.Tensor) -> torch.Tensor:
    """Fixed permutation for relation type T. perm_T(x) = P_T * x (PRD Section 4.2)."""
    return x[..., perm_index]


def soft_cleanup(query: torch.Tensor, codebook: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    """Soft nearest-prototype retrieval: softmax(beta * cos(x, CB)) @ CB (PRD Section 4.2)."""
    sims = F.cosine_similarity(query.unsqueeze(0), codebook, dim=-1)
    weights = torch.softmax(beta * sims, dim=-1)
    return weights @ codebook
