"""Tests for BondSystem — form/recover/decay cycle, fix #3 unbind math."""
import torch
import pytest
from psn2.bonds import BondSystem, PermutationIndex
from psn2.vsa import normalize, bind


def test_form_bond_legal():
    bs = BondSystem(dim=64)
    src = normalize(torch.randn(64))
    tgt = normalize(torch.randn(64))
    bond = bs.form_bond("causal", 0, 1, src, tgt, shape_type="compositional")
    assert bond is not None
    assert len(bs.bonds) == 1


def test_form_bond_illegal():
    bs = BondSystem(dim=64)
    src = normalize(torch.randn(64))
    tgt = normalize(torch.randn(64))
    # MODIFIES (type 10) not allowed in perceptive shape
    bond = bs.form_bond("MODIFIES", 0, 1, src, tgt, shape_type="perceptive")
    assert bond is None


def test_recover_source_normalized():
    """Fix #3: recover_source should work for normalized (non-bipolar) vectors."""
    dim = 512
    bs = BondSystem(dim=dim)
    # Use normalized continuous vectors (not bipolar)
    src = normalize(torch.randn(dim))
    tgt = normalize(torch.randn(dim))
    bond = bs.form_bond("causal", 0, 1, src, tgt, shape_type="compositional")
    assert bond is not None

    # Build a small codebook containing src (normalized)
    codebook = torch.stack([src] + [normalize(torch.randn(dim)) for _ in range(9)], dim=0)
    idx, recovered, sim = bs.recover_source(bond, tgt, codebook)
    # Should recover src (index 0) with high similarity
    assert idx == 0, f"Expected to recover src at index 0, got {idx} (sim={sim:.3f})"
    assert sim > 0.5, f"Recovery similarity too low: {sim:.3f}"


def test_bond_decay():
    bs = BondSystem(dim=64)
    src = normalize(torch.randn(64))
    tgt = normalize(torch.randn(64))
    bs.form_bond("causal", 0, 1, src, tgt)
    initial_strength = bs.bonds[0].strength
    bs.pulse_decay()
    assert bs.bonds[0].strength < initial_strength


def test_bond_decay_removes_dead():
    bs = BondSystem(dim=64)
    src = normalize(torch.randn(64))
    tgt = normalize(torch.randn(64))
    bs.form_bond("causal", 0, 1, src, tgt)
    bs.bonds[0].strength = 0.005  # below threshold
    bs.pulse_decay()
    assert len(bs.bonds) == 0


def test_state_dict_roundtrip():
    bs = BondSystem(dim=64)
    src = normalize(torch.randn(64))
    tgt = normalize(torch.randn(64))
    bs.form_bond("causal", 0, 1, src, tgt)
    state = bs.state_dict()
    bs2 = BondSystem(dim=64)
    bs2.load_state_dict(state)
    assert len(bs2.bonds) == 1
    assert bs2.bonds[0].bond_type == "causal"
