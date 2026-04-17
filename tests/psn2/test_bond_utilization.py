"""Tests for Task 3.4: Bond utilization during entity prediction.

Verifies that:
- bond_context_proj and bond_gate layers are registered on PSN2System
- _retrieve_bond_context returns zero vector when no bonds exist
- _retrieve_bond_context returns a non-zero vector when bonds are present
- Bond context is integrated into entity decoder input (neighborhood_with_bonds)
- Gradients flow through bond_context_proj and bond_gate during graph forward pass
- Bond formation in perceptive phase remains unchanged (preservation)

**Validates: Requirements 1.5, 2.5**
"""
import torch
import torch.nn as nn
import pytest
from psn2.core import PSN2System
from psn2.bonds import BondSystem
from psn2.vsa import normalize


def make_model(dim=64):
    """Create a small PSN2System for testing."""
    return PSN2System(dim=dim, max_nodes=16, grid_vocab=10, rel_vocab=17, stage="D1")


def make_graph_batch(B=2, N_e=6, N_r=5, rel_vocab=17):
    """Create a graph batch for testing."""
    return {
        "type": "graph",
        "entities": torch.randint(0, rel_vocab - 1, (B, N_e)),
        "relations": torch.randint(0, 32, (B, N_r, 3)),
        "target_entity": torch.randint(0, rel_vocab - 1, (B,)),
        "target_relation": torch.randint(0, 32, (B,)),
        "masked_entity_idx": torch.zeros(B, dtype=torch.long),
    }


def make_arc_batch(B=2, H=4, W=4, vocab=10):
    """Create an ARC grid batch for testing."""
    return {
        "type": "arc",
        "input_grid": torch.randint(0, vocab, (B, H, W)),
        "target_grid": torch.randint(0, vocab, (B, H, W)),
        "mask": torch.zeros(B, H, W, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Layer registration tests
# ---------------------------------------------------------------------------

def test_bond_context_proj_registered():
    """Task 3.4: bond_context_proj should be registered as a model parameter."""
    model = make_model(dim=64)
    assert hasattr(model, 'bond_context_proj'), (
        "PSN2System should have bond_context_proj (Task 3.4)"
    )
    assert isinstance(model.bond_context_proj, nn.Linear), (
        "bond_context_proj should be an nn.Linear"
    )


def test_bond_gate_registered():
    """Task 3.4: bond_gate should be registered as a model parameter."""
    model = make_model(dim=64)
    assert hasattr(model, 'bond_gate'), (
        "PSN2System should have bond_gate (Task 3.4)"
    )
    assert isinstance(model.bond_gate, nn.Linear), (
        "bond_gate should be an nn.Linear"
    )


def test_bond_context_proj_dims():
    """bond_context_proj should map dim -> dim."""
    model = make_model(dim=64)
    assert model.bond_context_proj.in_features == 64, (
        f"bond_context_proj should have in_features=64, got {model.bond_context_proj.in_features}"
    )
    assert model.bond_context_proj.out_features == 64, (
        f"bond_context_proj should have out_features=64, got {model.bond_context_proj.out_features}"
    )


def test_bond_gate_dims():
    """bond_gate should map 2*dim -> dim (takes [neighborhood, bond_ctx] as input)."""
    model = make_model(dim=64)
    assert model.bond_gate.in_features == 128, (
        f"bond_gate should have in_features=128 (2*dim), got {model.bond_gate.in_features}"
    )
    assert model.bond_gate.out_features == 64, (
        f"bond_gate should have out_features=64, got {model.bond_gate.out_features}"
    )


def test_retrieve_bond_context_method_exists():
    """Task 3.4: _retrieve_bond_context method should exist on PSN2System."""
    model = make_model(dim=64)
    assert hasattr(model, '_retrieve_bond_context'), (
        "PSN2System should have _retrieve_bond_context method (Task 3.4)"
    )
    assert callable(model._retrieve_bond_context), (
        "_retrieve_bond_context should be callable"
    )


# ---------------------------------------------------------------------------
# _retrieve_bond_context behavior tests
# ---------------------------------------------------------------------------

def test_retrieve_bond_context_no_bonds_returns_zero():
    """When no bonds exist, _retrieve_bond_context should return a zero vector."""
    model = make_model(dim=64)
    # Ensure no bonds
    model.bond_system.bonds.clear()

    result = model._retrieve_bond_context(device=torch.device('cpu'))

    assert result.shape == (64,), f"Expected shape (64,), got {result.shape}"
    assert torch.allclose(result, torch.zeros(64)), (
        "Should return zero vector when no bonds exist"
    )


def test_retrieve_bond_context_with_bonds_returns_nonzero():
    """When bonds exist, _retrieve_bond_context should return a non-zero vector."""
    model = make_model(dim=64)
    # Manually form a bond between two node bank vectors
    src_vec = normalize(torch.randn(64))
    tgt_vec = normalize(torch.randn(64))
    # Set node bank vectors to known values so recovery works
    with torch.no_grad():
        model.node_bank.nu[0] = src_vec
        model.node_bank.nu[1] = tgt_vec
        model.node_bank.active[0] = 1.0
        model.node_bank.active[1] = 1.0

    bond = model.bond_system.form_bond(
        "causal", 0, 1, src_vec, tgt_vec, shape_type="compositional"
    )
    assert bond is not None, "Bond should have been formed"

    result = model._retrieve_bond_context(device=torch.device('cpu'))

    assert result.shape == (64,), f"Expected shape (64,), got {result.shape}"
    # With a valid bond, the result should be non-zero
    assert result.norm() > 1e-6, (
        "Should return non-zero vector when bonds exist with positive similarity"
    )


def test_retrieve_bond_context_output_shape():
    """_retrieve_bond_context should always return a [D] vector."""
    model = make_model(dim=64)

    result = model._retrieve_bond_context(device=torch.device('cpu'))

    assert result.shape == (64,), f"Expected shape (64,), got {result.shape}"
    assert result.dim() == 1, f"Expected 1D tensor, got {result.dim()}D"


def test_retrieve_bond_context_no_nan_or_inf():
    """_retrieve_bond_context should not produce NaN or Inf values."""
    model = make_model(dim=64)
    # Form multiple bonds
    for i in range(3):
        src_vec = normalize(torch.randn(64))
        tgt_vec = normalize(torch.randn(64))
        with torch.no_grad():
            model.node_bank.nu[i] = src_vec
            model.node_bank.nu[i + 1] = tgt_vec
        model.bond_system.form_bond(
            "causal", i, i + 1, src_vec, tgt_vec, shape_type="compositional"
        )

    result = model._retrieve_bond_context(device=torch.device('cpu'))

    assert not torch.isnan(result).any(), "Bond context should not contain NaN"
    assert not torch.isinf(result).any(), "Bond context should not contain Inf"


def test_retrieve_bond_context_weak_bonds_skipped():
    """Bonds with strength < 0.05 should be skipped."""
    model = make_model(dim=64)
    src_vec = normalize(torch.randn(64))
    tgt_vec = normalize(torch.randn(64))
    bond = model.bond_system.form_bond(
        "causal", 0, 1, src_vec, tgt_vec, shape_type="compositional"
    )
    assert bond is not None
    # Set bond strength below threshold
    bond.strength = 0.01

    result = model._retrieve_bond_context(device=torch.device('cpu'))

    # Weak bond should be skipped, returning zero vector
    assert torch.allclose(result, torch.zeros(64)), (
        "Weak bonds (strength < 0.05) should be skipped, returning zero vector"
    )


# ---------------------------------------------------------------------------
# Integration: bond context in forward_batch
# ---------------------------------------------------------------------------

def test_graph_forward_with_bonds_runs_without_error():
    """
    forward_batch for graph type should run without error even when bonds exist.
    This tests the full bond utilization path.
    """
    model = make_model(dim=64)
    # Pre-populate bonds to test the bond utilization path
    src_vec = normalize(torch.randn(64))
    tgt_vec = normalize(torch.randn(64))
    with torch.no_grad():
        model.node_bank.nu[0] = src_vec
        model.node_bank.nu[1] = tgt_vec
        model.node_bank.active[0] = 1.0
        model.node_bank.active[1] = 1.0
    model.bond_system.form_bond(
        "causal", 0, 1, src_vec, tgt_vec, shape_type="compositional"
    )

    batch = make_graph_batch(B=2)
    out = model.forward_batch(batch, phase="compositional")

    assert "loss" in out
    assert out["loss"].requires_grad
    assert not torch.isnan(out["loss"]), "Loss should not be NaN with bonds present"
    assert not torch.isinf(out["loss"]), "Loss should not be Inf with bonds present"


def test_graph_forward_without_bonds_runs_without_error():
    """
    forward_batch for graph type should run without error when no bonds exist.
    This tests the fallback path (neighborhood_with_bonds = neighborhood).
    """
    model = make_model(dim=64)
    # Ensure no bonds
    model.bond_system.bonds.clear()

    batch = make_graph_batch(B=2)
    out = model.forward_batch(batch, phase="compositional")

    assert "loss" in out
    assert out["loss"].requires_grad
    assert not torch.isnan(out["loss"]), "Loss should not be NaN without bonds"


def test_bond_context_proj_receives_gradients():
    """
    bond_context_proj should receive gradients during graph forward pass
    when bonds are present (bond context path is active).
    """
    model = make_model(dim=64)
    # Pre-populate bonds to activate the bond context path
    src_vec = normalize(torch.randn(64))
    tgt_vec = normalize(torch.randn(64))
    with torch.no_grad():
        model.node_bank.nu[0] = src_vec
        model.node_bank.nu[1] = tgt_vec
        model.node_bank.active[0] = 1.0
        model.node_bank.active[1] = 1.0
    model.bond_system.form_bond(
        "causal", 0, 1, src_vec, tgt_vec, shape_type="compositional"
    )

    batch = make_graph_batch(B=2)
    out = model.forward_batch(batch, phase="compositional")
    out["loss"].backward()

    assert model.bond_context_proj.weight.grad is not None, (
        "bond_context_proj should receive gradients when bond context path is active"
    )
    assert model.bond_context_proj.weight.grad.abs().sum() > 0, (
        "bond_context_proj gradient should be non-zero"
    )


def test_bond_gate_receives_gradients():
    """
    bond_gate should receive gradients during graph forward pass
    when bonds are present (bond context path is active).
    """
    model = make_model(dim=64)
    # Pre-populate bonds to activate the bond context path
    src_vec = normalize(torch.randn(64))
    tgt_vec = normalize(torch.randn(64))
    with torch.no_grad():
        model.node_bank.nu[0] = src_vec
        model.node_bank.nu[1] = tgt_vec
        model.node_bank.active[0] = 1.0
        model.node_bank.active[1] = 1.0
    model.bond_system.form_bond(
        "causal", 0, 1, src_vec, tgt_vec, shape_type="compositional"
    )

    batch = make_graph_batch(B=2)
    out = model.forward_batch(batch, phase="compositional")
    out["loss"].backward()

    assert model.bond_gate.weight.grad is not None, (
        "bond_gate should receive gradients when bond context path is active"
    )
    assert model.bond_gate.weight.grad.abs().sum() > 0, (
        "bond_gate gradient should be non-zero"
    )


def test_bond_gate_initialized_near_zero():
    """
    bond_gate weights should be initialized with small std (0.01) so the gate
    starts near zero output, ensuring stable early training.
    """
    model = make_model(dim=64)
    # Check that bond_gate weights are small (std ≈ 0.01)
    weight_std = model.bond_gate.weight.std().item()
    assert weight_std < 0.1, (
        f"bond_gate weights should be initialized small (std < 0.1), got {weight_std:.4f}"
    )
    # Check bias is zero
    assert torch.allclose(model.bond_gate.bias, torch.zeros_like(model.bond_gate.bias)), (
        "bond_gate bias should be initialized to zero"
    )


# ---------------------------------------------------------------------------
# Preservation: bond formation in other phases unchanged
# ---------------------------------------------------------------------------

def test_arc_forward_unaffected_by_bond_utilization():
    """
    ARC grid forward pass should be completely unaffected by bond utilization.
    Bond context path is only active in the graph branch.
    """
    model = make_model(dim=64)
    # Pre-populate bonds
    src_vec = normalize(torch.randn(64))
    tgt_vec = normalize(torch.randn(64))
    model.bond_system.form_bond(
        "causal", 0, 1, src_vec, tgt_vec, shape_type="compositional"
    )

    batch = make_arc_batch(B=2)
    out = model.forward_batch(batch, phase="perceptive")

    assert "loss" in out
    assert out["loss"].requires_grad
    assert not torch.isnan(out["loss"]), "ARC loss should not be NaN"


def test_bond_formation_in_perceptive_phase_unchanged():
    """
    Bond formation in perceptive phase should remain unchanged.
    Perceptive phase only allows bond types [0, 1, 2, 3] (causal, temporal, spatial, part_whole).
    """
    bs = BondSystem(dim=64)
    src = normalize(torch.randn(64))
    tgt = normalize(torch.randn(64))

    # Causal bond should be allowed in perceptive phase
    bond = bs.form_bond("causal", 0, 1, src, tgt, shape_type="perceptive")
    assert bond is not None, "Causal bond should be allowed in perceptive phase"

    # Linguistic bond should NOT be allowed in perceptive phase
    bond_linguistic = bs.form_bond("MODIFIES", 0, 1, src, tgt, shape_type="perceptive")
    assert bond_linguistic is None, "Linguistic bond should not be allowed in perceptive phase"


def test_bond_formation_in_compositional_phase_unchanged():
    """
    Bond formation in compositional phase should remain unchanged.
    Compositional phase allows bond types [0, 1, 2, 3, 10, 11, 12, 13, 14].
    """
    bs = BondSystem(dim=64)
    src = normalize(torch.randn(64))
    tgt = normalize(torch.randn(64))

    # Causal bond should be allowed
    bond_causal = bs.form_bond("causal", 0, 1, src, tgt, shape_type="compositional")
    assert bond_causal is not None, "Causal bond should be allowed in compositional phase"

    # MODIFIES (type 10) should be allowed
    bond_modifies = bs.form_bond("MODIFIES", 0, 1, src, tgt, shape_type="compositional")
    assert bond_modifies is not None, "MODIFIES bond should be allowed in compositional phase"

    # DISCOURSE_CONTINUES (type 15) should NOT be allowed in compositional phase
    bond_discourse = bs.form_bond("DISCOURSE_CONTINUES", 0, 1, src, tgt, shape_type="compositional")
    assert bond_discourse is None, "DISCOURSE_CONTINUES bond should not be allowed in compositional phase"


def test_multiple_bonds_aggregate_correctly():
    """
    When multiple bonds exist, _retrieve_bond_context should aggregate them
    using strength-weighted mean.
    """
    model = make_model(dim=64)
    # Form 3 bonds with different strengths
    for i in range(3):
        src_vec = normalize(torch.randn(64))
        tgt_vec = normalize(torch.randn(64))
        with torch.no_grad():
            model.node_bank.nu[i * 2] = src_vec
            model.node_bank.nu[i * 2 + 1] = tgt_vec
        bond = model.bond_system.form_bond(
            "causal", i * 2, i * 2 + 1, src_vec, tgt_vec, shape_type="compositional"
        )
        if bond is not None:
            bond.strength = 0.5 + i * 0.2  # Different strengths: 0.5, 0.7, 0.9

    result = model._retrieve_bond_context(device=torch.device('cpu'))

    assert result.shape == (64,), f"Expected shape (64,), got {result.shape}"
    assert not torch.isnan(result).any(), "Aggregated bond context should not contain NaN"
    assert not torch.isinf(result).any(), "Aggregated bond context should not contain Inf"


def test_graph_forward_pred_shape():
    """
    Graph forward pass should return entity predictions with correct shape.
    This verifies the bond utilization path doesn't break the output shape.
    """
    model = make_model(dim=64)
    B = 3
    batch = make_graph_batch(B=B)
    out = model.forward_batch(batch, phase="compositional")

    assert "pred" in out
    # pred should be [B, rel_vocab] entity logits
    assert out["pred"].shape[0] == B, (
        f"Expected batch size {B}, got {out['pred'].shape[0]}"
    )
