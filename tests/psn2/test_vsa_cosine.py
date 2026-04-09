"""Tests for vsa.cosine() batched correctness fix (#16)."""
import torch
import pytest
from psn2.vsa import cosine, normalize


def test_cosine_1d():
    a = normalize(torch.randn(512))
    b = normalize(torch.randn(512))
    sim = cosine(a, b)
    assert sim.shape == torch.Size([]), "1D cosine should return scalar"
    assert -1.0 <= float(sim) <= 1.0


def test_cosine_identical_1d():
    a = normalize(torch.randn(512))
    sim = cosine(a, a)
    assert abs(float(sim) - 1.0) < 1e-5


def test_cosine_batched():
    """Fix #16: batched cosine must not flatten to [B*D] and give wrong result."""
    B, D = 4, 512
    a = normalize(torch.randn(B, D))
    b = normalize(torch.randn(B, D))
    sim = cosine(a, b)
    # Should return a scalar (mean of per-sample similarities), not a single
    # wrong value from flattening [B*D] vectors
    assert sim.shape == torch.Size([]), f"Batched cosine should return scalar mean, got shape {sim.shape}"
    assert -1.0 <= float(sim) <= 1.0


def test_cosine_batched_identical():
    """Batched cosine of identical tensors should be ~1.0."""
    B, D = 4, 512
    a = normalize(torch.randn(B, D))
    sim = cosine(a, a)
    assert abs(float(sim) - 1.0) < 1e-5, f"Identical batched cosine should be 1.0, got {float(sim)}"


def test_cosine_batched_vs_1d_consistent():
    """Mean of per-sample cosines should match batched cosine."""
    B, D = 4, 64
    a = normalize(torch.randn(B, D))
    b = normalize(torch.randn(B, D))
    import torch.nn.functional as F
    per_sample = F.cosine_similarity(a, b, dim=-1).mean()
    batched = cosine(a, b)
    assert abs(float(per_sample) - float(batched)) < 1e-5
