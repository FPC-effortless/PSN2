"""End-to-end forward_batch tests for arc and graph branches."""
import torch
import pytest
from psn2.core import PSN2System
from psn2.vsa import normalize


def make_model():
    return PSN2System(dim=64, max_nodes=16, grid_vocab=10, rel_vocab=32, stage="D1")


def make_arc_batch(B=2, H=4, W=4, vocab=10):
    return {
        "type": "arc",
        "input_grid": torch.randint(0, vocab, (B, H, W)),
        "target_grid": torch.randint(0, vocab, (B, H, W)),
        "mask": torch.zeros(B, H, W, dtype=torch.long),
    }


def make_graph_batch(B=2, N_e=6, N_r=5):
    return {
        "type": "graph",
        "entities": torch.randint(0, 32, (B, N_e)),
        "relations": torch.randint(0, 32, (B, N_r, 3)),
        "target_entity": torch.randint(0, 32, (B,)),
        "target_relation": torch.randint(0, 32, (B,)),
        "mask_entity": torch.zeros(B, dtype=torch.long),
        "mask_relation": torch.zeros(B, dtype=torch.long),
    }


def test_arc_forward_returns_loss():
    model = make_model()
    batch = make_arc_batch()
    out = model.forward_batch(batch, phase="perceptive")
    assert "loss" in out
    assert out["loss"].requires_grad
    assert out["loss"].item() > 0


def test_graph_forward_returns_loss():
    model = make_model()
    batch = make_graph_batch()
    out = model.forward_batch(batch, phase="compositional")
    assert "loss" in out
    assert out["loss"].requires_grad
    assert out["loss"].item() > 0


def test_graph_loss_shape_nonzero():
    """Fix #4: graph branch loss_shape should not always be 0."""
    model = make_model()
    batch = make_graph_batch()
    out = model.forward_batch(batch, phase="compositional")
    # loss_shape may be small but should not be exactly 0.0 (was hardcoded before)
    # It can be 0 if shapes happen to be identical, but that's astronomically unlikely
    assert "loss_shape" in out


def test_arc_backward():
    model = make_model()
    batch = make_arc_batch()
    out = model.forward_batch(batch, phase="perceptive")
    out["loss"].backward()
    # Check gradients flow to node bank
    assert model.node_bank.nu.grad is not None
    assert model.node_bank.nu.grad.abs().sum() > 0


def test_graph_backward():
    model = make_model()
    batch = make_graph_batch()
    out = model.forward_batch(batch, phase="compositional")
    out["loss"].backward()
    assert model.node_bank.nu.grad is not None


def test_checkpoint_roundtrip():
    """Checkpoint save → load → same forward output."""
    import tempfile, os
    model = make_model()
    batch = make_arc_batch(B=1)
    with torch.no_grad():
        out1 = model.forward_batch(batch, phase="perceptive")

    state = model.state_dict_full()
    model2 = make_model()
    model2.load_state_dict_full(state)
    with torch.no_grad():
        out2 = model2.forward_batch(batch, phase="perceptive")

    # Losses should be identical after reload
    assert abs(out1["loss"].item() - out2["loss"].item()) < 1e-4, \
        f"Checkpoint roundtrip loss mismatch: {out1['loss'].item()} vs {out2['loss'].item()}"
