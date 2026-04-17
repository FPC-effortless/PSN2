"""Preservation property tests for D1 relation prediction bugfix.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

These tests verify that the bugfix does NOT regress existing functionality:
- Grid-based task performance (80%+ object tracking) must continue to work
- Perceptive phase processing must remain unchanged
- Other D1 gates must continue to pass
- Mixed batch training must not degrade ARC performance

**IMPORTANT**: These tests follow observation-first methodology:
1. Run tests on UNFIXED code to observe baseline behavior
2. Tests should PASS on unfixed code (confirming baseline to preserve)
3. After implementing fix, re-run tests to ensure they still PASS (no regressions)
"""
import torch
import pytest
from hypothesis import given, strategies as st, settings, Phase
from psn2.core import PSN2System
from psn2.datasets.arc_grid import ARCGridDataset


def make_arc_batch(B=4, grid_size=8, vocab=10):
    """Create an ARC grid batch for testing."""
    dataset = ARCGridDataset(n_samples=B, grid_size=grid_size, vocab=vocab)
    batches = [dataset[i] for i in range(B)]
    
    # Stack into batch
    return {
        "type": "arc",
        "input_grid": torch.stack([b["input_grid"] for b in batches]),
        "target_grid": torch.stack([b["target_grid"] for b in batches]),
        "mask": torch.stack([b["mask"] for b in batches]),
    }


@pytest.mark.slow
def test_preservation_grid_task_object_tracking():
    """
    **Property 2: Preservation** - Grid Task Object Tracking Accuracy >= 80%
    
    **Validates: Requirements 3.1, 3.2**
    
    **EXPECTED OUTCOME**: Test PASSES on unfixed code (confirms baseline behavior)
    
    This test verifies that ARC grid batch processing during perceptive phase
    maintains object tracking accuracy of 80%+ after training. This is the
    baseline behavior that must be preserved after the bugfix.
    """
    # Initialize model
    torch.manual_seed(42)
    model = PSN2System(dim=128, max_nodes=32, grid_vocab=10, rel_vocab=64, stage="D1")
    model.train()
    
    # Use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training configuration
    num_steps = 2000
    batch_size = 4
    grid_size = 8
    vocab = 10
    
    # Metrics tracking
    correct_predictions = 0
    total_predictions = 0
    
    print(f"\n{'='*70}")
    print(f"Preservation Test: Grid Task Object Tracking - {num_steps} steps")
    print(f"{'='*70}")
    
    # Training loop on ARC grid batches
    for step in range(num_steps):
        # Generate ARC grid batch
        batch = make_arc_batch(B=batch_size, grid_size=grid_size, vocab=vocab)
        
        # Forward pass in PERCEPTIVE phase (grid tasks use perceptive phase)
        optimizer.zero_grad()
        output = model.forward_batch(batch, phase="perceptive")
        loss = output["loss"]
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track accuracy (on ALL cells, matching evaluate.py metric)
        with torch.no_grad():
            pred_grids = output["pred"].argmax(dim=-1)  # [B, H, W]
            target_grids = batch["target_grid"]  # [B, H, W]
            
            # Evaluate on all cells (not just masked cells)
            correct = (pred_grids == target_grids).sum().item()
            total = target_grids.numel()
            
            correct_predictions += correct
            total_predictions += total
        
        # Log progress every 500 steps
        if (step + 1) % 500 == 0:
            current_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            print(f"Step {step + 1:5d} | Loss: {loss.item():.4f} | Accuracy: {current_accuracy:.4f}")
    
    # Final metrics
    final_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    print(f"\n{'='*70}")
    print(f"Final Metrics After {num_steps} Steps:")
    print(f"{'='*70}")
    print(f"Object Tracking Accuracy: {final_accuracy:.4f} (target: >= 0.80)")
    print(f"{'='*70}")
    
    # This test should PASS on unfixed code (baseline behavior)
    # After the fix, it should still PASS (no regression)
    assert final_accuracy >= 0.80, (
        f"Object tracking accuracy {final_accuracy:.4f} < 0.80. "
        f"Expected: >= 80% accuracy on ARC grid batches. "
        f"This indicates a regression in grid task performance."
    )
    
    print(f"\n✅ PRESERVATION TEST PASSED: Grid task performance maintained at {final_accuracy:.4f}")


@pytest.mark.slow
def test_preservation_perceptive_phase_committed_shape_stability():
    """
    **Property 2: Preservation** - Perceptive Phase Committed Shape Stability
    
    **Validates: Requirements 3.2**
    
    **EXPECTED OUTCOME**: Test PASSES on unfixed code (confirms baseline behavior)
    
    This test verifies that the perceptive phase processing produces stable
    committed_shape outputs for grid inputs. The shape should converge to
    a consistent representation after multiple pulses.
    """
    torch.manual_seed(42)
    model = PSN2System(dim=128, max_nodes=32, grid_vocab=10, rel_vocab=64, stage="D1")
    model.eval()
    
    # Generate a fixed ARC grid batch
    batch = make_arc_batch(B=4, grid_size=8, vocab=10)
    
    print(f"\n{'='*70}")
    print(f"Preservation Test: Perceptive Phase Committed Shape Stability")
    print(f"{'='*70}")
    
    # Run multiple forward passes and collect shapes
    shapes = []
    with torch.no_grad():
        for i in range(10):
            output = model.forward_batch(batch, phase="perceptive")
            shape = output["shape"]  # [B, D]
            shapes.append(shape)
    
    # Compute pairwise cosine similarities between shapes
    similarities = []
    for i in range(len(shapes) - 1):
        sim = torch.nn.functional.cosine_similarity(shapes[i], shapes[i+1], dim=-1).mean().item()
        similarities.append(sim)
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    
    print(f"Average pairwise cosine similarity: {avg_similarity:.4f} (target: >= 0.95)")
    print(f"{'='*70}")
    
    # Committed shape should be stable across runs (high cosine similarity)
    assert avg_similarity >= 0.95, (
        f"Average shape similarity {avg_similarity:.4f} < 0.95. "
        f"Expected: committed_shape should be stable across multiple forward passes. "
        f"This indicates a regression in perceptive phase processing."
    )
    
    print(f"\n✅ PRESERVATION TEST PASSED: Committed shape stability at {avg_similarity:.4f}")


@pytest.mark.slow
def test_preservation_loss_computation_consistency():
    """
    **Property 2: Preservation** - Loss Computation Consistency for Grid Tasks
    
    **Validates: Requirements 3.5**
    
    **EXPECTED OUTCOME**: Test PASSES on unfixed code (confirms baseline behavior)
    
    This test verifies that the loss computation for grid batches remains
    consistent. The loss should decrease during training and maintain the
    same component structure (L_error, L_shape, L_vsa, L_compact).
    """
    torch.manual_seed(42)
    model = PSN2System(dim=128, max_nodes=32, grid_vocab=10, rel_vocab=64, stage="D1")
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    num_steps = 1000
    batch_size = 4
    
    print(f"\n{'='*70}")
    print(f"Preservation Test: Loss Computation Consistency - {num_steps} steps")
    print(f"{'='*70}")
    
    initial_loss = None
    final_loss = None
    
    for step in range(num_steps):
        batch = make_arc_batch(B=batch_size, grid_size=8, vocab=10)
        
        optimizer.zero_grad()
        output = model.forward_batch(batch, phase="perceptive")
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        
        if step == 0:
            initial_loss = loss.item()
        if step == num_steps - 1:
            final_loss = loss.item()
        
        if (step + 1) % 250 == 0:
            print(f"Step {step + 1:5d} | Loss: {loss.item():.4f}")
    
    loss_reduction = (initial_loss - final_loss) / initial_loss
    
    print(f"\n{'='*70}")
    print(f"Loss Metrics:")
    print(f"{'='*70}")
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss:   {final_loss:.4f}")
    print(f"Reduction:    {loss_reduction:.2%} (target: >= 20%)")
    print(f"{'='*70}")
    
    # Loss should decrease by at least 20% during training
    assert loss_reduction >= 0.20, (
        f"Loss reduction {loss_reduction:.2%} < 20%. "
        f"Expected: loss should decrease by at least 20% during training. "
        f"This indicates a regression in loss computation or training dynamics."
    )
    
    # Verify loss components exist in output
    assert "loss_pred" in output, "Missing loss_pred component"
    assert "loss_shape" in output, "Missing loss_shape component"
    assert "loss_spike" in output, "Missing loss_spike component"
    
    print(f"\n✅ PRESERVATION TEST PASSED: Loss computation consistent, reduction {loss_reduction:.2%}")


@pytest.mark.slow
@given(
    grid_size=st.integers(min_value=6, max_value=10),
    vocab=st.integers(min_value=8, max_value=12),
)
@settings(
    max_examples=3,
    phases=[Phase.generate, Phase.target],
    deadline=None,
)
def test_preservation_property_based_grid_accuracy(grid_size, vocab):
    """
    **Property 2: Preservation (Property-Based)** - Grid Accuracy Across Configurations
    
    **Validates: Requirements 3.1, 3.4**
    
    **EXPECTED OUTCOME**: Test PASSES on unfixed code (confirms baseline behavior)
    
    Property-based test that generates different grid configurations to verify
    that grid task performance is preserved across various input sizes and
    vocabulary sizes.
    """
    torch.manual_seed(42)
    model = PSN2System(dim=128, max_nodes=32, grid_vocab=vocab, rel_vocab=64, stage="D1")
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Shorter training for property-based test
    num_steps = 500
    batch_size = 4
    
    correct_predictions = 0
    total_predictions = 0
    
    for step in range(num_steps):
        batch = make_arc_batch(B=batch_size, grid_size=grid_size, vocab=vocab)
        
        optimizer.zero_grad()
        output = model.forward_batch(batch, phase="perceptive")
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            pred_grids = output["pred"].argmax(dim=-1)
            target_grids = batch["target_grid"]
            
            correct = (pred_grids == target_grids).sum().item()
            total = target_grids.numel()
            
            correct_predictions += correct
            total_predictions += total
    
    final_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # For property-based test, we expect at least 60% accuracy after 500 steps
    # (lower threshold due to shorter training)
    assert final_accuracy >= 0.60, (
        f"Grid accuracy {final_accuracy:.4f} < 0.60 after {num_steps} steps "
        f"with grid_size={grid_size}, vocab={vocab}. "
        f"Expected: >= 60% accuracy on grid tasks. "
        f"This indicates a regression in grid task performance."
    )


@pytest.mark.slow
def test_preservation_mixed_batch_training():
    """
    **Property 2: Preservation** - Mixed Batch Training (60% ARC, 40% Graph)
    
    **Validates: Requirements 3.4**
    
    **EXPECTED OUTCOME**: Test PASSES on unfixed code (confirms baseline behavior)
    
    This test verifies that training on mixed batches (60% ARC grids, 40% graphs)
    does not degrade ARC performance. The grid task accuracy should remain high
    even when training includes graph batches.
    """
    torch.manual_seed(42)
    model = PSN2System(dim=128, max_nodes=32, grid_vocab=10, rel_vocab=64, stage="D1")
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    num_steps = 1000
    batch_size = 4
    
    # Track ARC-only accuracy
    arc_correct = 0
    arc_total = 0
    
    print(f"\n{'='*70}")
    print(f"Preservation Test: Mixed Batch Training - {num_steps} steps")
    print(f"{'='*70}")
    
    for step in range(num_steps):
        # 60% ARC, 40% graph mix
        if torch.rand(1).item() < 0.60:
            # ARC batch
            batch = make_arc_batch(B=batch_size, grid_size=8, vocab=10)
            phase = "perceptive"
            
            optimizer.zero_grad()
            output = model.forward_batch(batch, phase=phase)
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            
            # Track ARC accuracy (on all cells)
            with torch.no_grad():
                pred_grids = output["pred"].argmax(dim=-1)
                target_grids = batch["target_grid"]
                
                correct = (pred_grids == target_grids).sum().item()
                total = target_grids.numel()
                
                arc_correct += correct
                arc_total += total
        else:
            # Graph batch (from bug condition test)
            batch = {
                "type": "graph",
                "entities": torch.randint(0, 64, (batch_size, 6)),
                "relations": torch.randint(0, 32, (batch_size, 5, 3)),
                "target_entity": torch.randint(0, 64, (batch_size,)),
                "target_relation": torch.randint(0, 32, (batch_size,)),
                "masked_entity_idx": torch.randint(0, 6, (batch_size,)),
            }
            phase = "compositional"
            
            optimizer.zero_grad()
            output = model.forward_batch(batch, phase=phase)
            loss = output["loss"]
            loss.backward()
            optimizer.step()
        
        if (step + 1) % 250 == 0:
            current_arc_accuracy = arc_correct / arc_total if arc_total > 0 else 0.0
            print(f"Step {step + 1:5d} | ARC Accuracy: {current_arc_accuracy:.4f}")
    
    final_arc_accuracy = arc_correct / arc_total if arc_total > 0 else 0.0
    
    print(f"\n{'='*70}")
    print(f"Final Metrics After {num_steps} Mixed Batch Steps:")
    print(f"{'='*70}")
    print(f"ARC Grid Accuracy: {final_arc_accuracy:.4f} (target: >= 0.75)")
    print(f"{'='*70}")
    
    # ARC accuracy should remain >= 75% even with mixed training
    assert final_arc_accuracy >= 0.75, (
        f"ARC accuracy {final_arc_accuracy:.4f} < 0.75 in mixed batch training. "
        f"Expected: >= 75% accuracy on ARC grids even with 40% graph batches. "
        f"This indicates that mixed batch training degrades ARC performance."
    )
    
    print(f"\n✅ PRESERVATION TEST PASSED: Mixed batch ARC accuracy at {final_arc_accuracy:.4f}")


@pytest.mark.slow
def test_preservation_d1_gates_continue_to_pass():
    """
    **Property 2: Preservation** - Other D1 Gates Continue to Pass
    
    **Validates: Requirements 3.3**
    
    **EXPECTED OUTCOME**: Test PASSES on unfixed code (confirms baseline behavior)
    
    This test verifies that other D1 gates (temporal_trace_persistence,
    vsa_binding_accuracy) continue to pass after training. The bugfix should
    not affect these metrics.
    """
    torch.manual_seed(42)
    model = PSN2System(dim=128, max_nodes=32, grid_vocab=10, rel_vocab=64, stage="D1")
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    num_steps = 1000
    batch_size = 4
    
    print(f"\n{'='*70}")
    print(f"Preservation Test: D1 Gates - {num_steps} steps")
    print(f"{'='*70}")
    
    # Train on ARC batches
    for step in range(num_steps):
        batch = make_arc_batch(B=batch_size, grid_size=8, vocab=10)
        
        optimizer.zero_grad()
        output = model.forward_batch(batch, phase="perceptive")
        loss = output["loss"]
        loss.backward()
        optimizer.step()
    
    # Check temporal trace persistence (node ages should accumulate)
    if hasattr(model.node_bank, 'ages'):
        max_age = model.node_bank.ages.max().item()
        print(f"Max node age: {max_age:.1f} (target: > 5.0)")
        
        assert max_age > 5.0, (
            f"Max node age {max_age:.1f} <= 5.0. "
            f"Expected: temporal trace persistence > 5 pulses. "
            f"This indicates a regression in temporal trace tracking."
        )
    
    # Check VSA binding accuracy (bond system should be functional)
    if hasattr(model.node_bank, 'bond_system') and model.node_bank.bond_system is not None:
        bond_system = model.node_bank.bond_system
        
        # Test VSA binding: bind two random vectors and recover
        test_source = torch.randn(model.dim)
        test_target = torch.randn(model.dim)
        
        # Bind using circular convolution
        bound = bond_system.bind(test_source, test_target)
        
        # Recover source from bound and target
        recovered = bond_system.unbind(bound, test_target)
        
        # Compute recovery accuracy (cosine similarity)
        recovery_accuracy = torch.nn.functional.cosine_similarity(
            test_source.unsqueeze(0), recovered.unsqueeze(0)
        ).item()
        
        print(f"VSA binding recovery accuracy: {recovery_accuracy:.4f} (target: > 0.75)")
        
        assert recovery_accuracy > 0.75, (
            f"VSA binding recovery accuracy {recovery_accuracy:.4f} <= 0.75. "
            f"Expected: VSA binding accuracy > 0.75. "
            f"This indicates a regression in VSA binding functionality."
        )
    
    print(f"\n{'='*70}")
    print(f"✅ PRESERVATION TEST PASSED: D1 gates continue to pass")
    print(f"{'='*70}")
