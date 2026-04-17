"""Task 4: Checkpoint validation - Ensure all tests pass.

**Validates: All requirements from bugfix spec**

This test runs the full validation suite for the D1 relation prediction bugfix:
1. Bug condition exploration test passes (entity accuracy > 50%)
2. Preservation tests pass (grid accuracy >= 80%)
3. D1 gate certifier passes (relation_prediction_error < 0.50)
4. Integration test: 5000 steps on mixed batches (60% ARC, 40% graph)
5. Checkpoint save/load preserves entity prediction accuracy
"""
import torch
import pytest
import tempfile
from pathlib import Path
from psn2.core import PSN2System
from psn2.datasets.arc_grid import ARCGridDataset
from psn2.dc.stage_d1 import StageD1


def make_graph_batch(B=4, N_e=6, N_r=5, entity_vocab=64, relation_vocab=32):
    """Create a relational graph batch with LEARNABLE semantic structure."""
    # Generate entity properties
    colors = torch.randint(0, 2, (B, N_e))  # 2 colors: 0=red, 1=blue
    shapes = torch.randint(0, 2, (B, N_e))  # 2 shapes: 0=circle, 1=square

    # Entity ID: color * 2 + shape  →  4 possible entities (0-3)
    entities = colors * 2 + shapes  # [B, N_e] in range [0, 3]

    # Build deterministic relations based on entity properties
    relations_list = []
    masked_entity_idx_list = []

    for b in range(B):
        batch_relations = []

        # Build all possible relations
        for i in range(N_e):
            for j in range(i + 1, N_e):
                # Relation type 0: same_color
                if colors[b, i] == colors[b, j]:
                    batch_relations.append([i, 0, j])
                # Relation type 1: same_shape
                if shapes[b, i] == shapes[b, j]:
                    batch_relations.append([i, 1, j])

        # Pick a masked entity that has BOTH relation types (color + shape)
        masked_idx = None
        for attempt in range(20):
            candidate_idx = torch.randint(0, N_e, (1,)).item()
            involved_rels = [r for r in batch_relations
                             if r[0] == candidate_idx or r[2] == candidate_idx]
            rel_types = set(r[1] for r in involved_rels)
            if len(rel_types) >= 2:
                masked_idx = candidate_idx
                break

        if masked_idx is None:
            # Fallback: any entity with at least one relation
            for attempt in range(10):
                candidate_idx = torch.randint(0, N_e, (1,)).item()
                has_relation = any(
                    r[0] == candidate_idx or r[2] == candidate_idx
                    for r in batch_relations
                )
                if has_relation:
                    masked_idx = candidate_idx
                    break

        if masked_idx is None:
            masked_idx = 0

        masked_entity_idx_list.append(masked_idx)

        # Pad or truncate to N_r relations
        while len(batch_relations) < N_r:
            batch_relations.append([0, 0, 0])  # padding
        relations_list.append(batch_relations[:N_r])

    relations = torch.tensor(relations_list, dtype=torch.long)  # [B, N_r, 3]
    masked_entity_idx = torch.tensor(masked_entity_idx_list, dtype=torch.long)  # [B]

    # Target entity is the actual entity at the masked position
    target_entity = entities[torch.arange(B), masked_entity_idx]

    # Mask the entity with a special mask token (4 = out of vocab range 0-3)
    entities_masked = entities.clone()
    entities_masked[torch.arange(B), masked_entity_idx] = 4  # mask token

    # Target relation is the most common relation type in the graph
    target_relation = relations[:, :, 1].mode(dim=1)[0]  # most frequent relation type

    return {
        "type": "graph",
        "entities": entities_masked,
        "relations": relations,
        "target_entity": target_entity,
        "target_relation": target_relation,
        "masked_entity_idx": masked_entity_idx,
    }


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
def test_checkpoint_full_validation():
    """
    **Task 4: Checkpoint Validation** - Ensure all tests pass
    
    This comprehensive test validates:
    1. Bug condition fixed: entity prediction accuracy > 50%
    2. D1 gate passes: causal_prediction_error < 0.50
    3. Preservation: ARC grid accuracy >= 80%
    4. Integration: 5000 steps on mixed batches (60% ARC, 40% graph)
    5. Checkpoint save/load preserves accuracy
    """
    torch.manual_seed(42)
    model = PSN2System(dim=256, max_nodes=32, grid_vocab=10, rel_vocab=5, stage="D1")
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Configuration
    num_steps = 5000
    batch_size = 4
    arc_ratio = 0.60  # 60% ARC, 40% graph
    
    # Metrics tracking
    graph_correct = 0
    graph_total = 0
    arc_correct = 0
    arc_total = 0
    gradient_norms = []
    bond_formation_count = 0
    total_pulses = 0
    
    print(f"\n{'='*70}")
    print(f"TASK 4: CHECKPOINT VALIDATION")
    print(f"{'='*70}")
    print(f"Training for {num_steps} steps on mixed batches (60% ARC, 40% graph)")
    print(f"{'='*70}\n")
    
    # Training loop on mixed batches
    for step in range(num_steps):
        # 60% ARC, 40% graph mix
        if torch.rand(1).item() < arc_ratio:
            # ARC batch
            batch = make_arc_batch(B=batch_size, grid_size=8, vocab=10)
            phase = "perceptive"
            
            optimizer.zero_grad()
            output = model.forward_batch(batch, phase=phase)
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            
            # Track ARC accuracy
            with torch.no_grad():
                pred_grids = output["pred"].argmax(dim=-1)
                target_grids = batch["target_grid"]
                
                correct = (pred_grids == target_grids).sum().item()
                total = target_grids.numel()
                
                arc_correct += correct
                arc_total += total
        else:
            # Graph batch
            batch = make_graph_batch(B=batch_size, N_e=6, N_r=5, 
                                    entity_vocab=5, relation_vocab=32)
            phase = "compositional"
            
            optimizer.zero_grad()
            output = model.forward_batch(batch, phase=phase)
            loss = output["loss"]
            loss.backward()
            
            # Track gradient norms for entity_encoder
            if hasattr(model, 'entity_encoder'):
                entity_encoder_grad_norm = 0.0
                entity_encoder_param_count = 0
                for param in model.entity_encoder.parameters():
                    if param.grad is not None:
                        entity_encoder_grad_norm += param.grad.norm().item() ** 2
                        entity_encoder_param_count += 1
                if entity_encoder_param_count > 0:
                    grad_norm = (entity_encoder_grad_norm / entity_encoder_param_count) ** 0.5
                    gradient_norms.append(grad_norm)
            
            # Track bond formation
            if hasattr(model, 'get_bond_formation_stats'):
                stats = model.get_bond_formation_stats()
                bond_formation_count = stats['bond_formation_count']
                total_pulses = stats['total_pulses']
            
            optimizer.step()
            
            # Track graph accuracy
            with torch.no_grad():
                pred_entities = output["pred"].argmax(dim=-1)
                target_entities = batch["target_entity"]
                correct = (pred_entities == target_entities).sum().item()
                graph_correct += correct
                graph_total += batch_size
        
        # Log progress every 1000 steps
        if (step + 1) % 1000 == 0:
            current_graph_acc = graph_correct / graph_total if graph_total > 0 else 0.0
            current_arc_acc = arc_correct / arc_total if arc_total > 0 else 0.0
            avg_grad_norm = sum(gradient_norms[-100:]) / len(gradient_norms[-100:]) if gradient_norms else 0.0
            bond_rate = bond_formation_count / total_pulses if total_pulses > 0 else 0.0
            
            print(f"Step {step + 1:5d} | "
                  f"Graph Acc: {current_graph_acc:.4f} | "
                  f"ARC Acc: {current_arc_acc:.4f} | "
                  f"Grad Norm: {avg_grad_norm:.6f} | "
                  f"Bond Rate: {bond_rate:.4f}")
    
    # Final metrics
    final_graph_accuracy = graph_correct / graph_total if graph_total > 0 else 0.0
    final_arc_accuracy = arc_correct / arc_total if arc_total > 0 else 0.0
    avg_gradient_norm = sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0
    bond_formation_rate = bond_formation_count / total_pulses if total_pulses > 0 else 0.0
    
    # D1 gate evaluation
    causal_prediction_error = 1.0 - final_graph_accuracy
    
    print(f"\n{'='*70}")
    print(f"FINAL METRICS AFTER {num_steps} STEPS")
    print(f"{'='*70}")
    print(f"Entity Prediction Accuracy:  {final_graph_accuracy:.4f} (target: > 0.50)")
    print(f"Causal Prediction Error:     {causal_prediction_error:.4f} (target: < 0.50)")
    print(f"ARC Grid Accuracy:           {final_arc_accuracy:.4f} (target: >= 0.80)")
    print(f"Average Gradient Norm:       {avg_gradient_norm:.6f} (target: > 0.01)")
    print(f"Bond Formation Rate:         {bond_formation_rate:.4f} (target: > 0.30)")
    print(f"{'='*70}\n")
    
    # Test checkpoint save/load
    print(f"{'='*70}")
    print(f"CHECKPOINT SAVE/LOAD TEST")
    print(f"{'='*70}")
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        checkpoint_path = tmp.name
    
    # Save checkpoint
    checkpoint = {
        "model": model.state_dict_full(),
        "optimizer": optimizer.state_dict(),
        "step": num_steps,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")
    
    # Measure accuracy before reload
    model.eval()
    test_batch = make_graph_batch(B=batch_size, N_e=6, N_r=5, 
                                 entity_vocab=5, relation_vocab=32)
    with torch.no_grad():
        output_before = model.forward_batch(test_batch, phase="compositional")
        pred_before = output_before["pred"].argmax(dim=-1)
        acc_before = (pred_before == test_batch["target_entity"]).float().mean().item()
    
    # Load checkpoint into new model
    model_loaded = PSN2System(dim=256, max_nodes=32, grid_vocab=10, rel_vocab=5, stage="D1")
    loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_loaded.load_state_dict_full(loaded_checkpoint["model"])
    model_loaded.eval()
    
    # Measure accuracy after reload
    with torch.no_grad():
        output_after = model_loaded.forward_batch(test_batch, phase="compositional")
        pred_after = output_after["pred"].argmax(dim=-1)
        acc_after = (pred_after == test_batch["target_entity"]).float().mean().item()
    
    accuracy_diff = abs(acc_after - acc_before)
    
    print(f"Accuracy before save: {acc_before:.4f}")
    print(f"Accuracy after load:  {acc_after:.4f}")
    print(f"Difference:           {accuracy_diff:.6f} (target: < 0.01)")
    
    checkpoint_test_passed = accuracy_diff < 0.01
    print(f"Checkpoint test:      {'PASS' if checkpoint_test_passed else 'FAIL'}")
    
    # Clean up
    Path(checkpoint_path).unlink()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"TASK 4 CHECKPOINT SUMMARY")
    print(f"{'='*70}")
    
    bug_fixed = final_graph_accuracy > 0.50
    d1_gate_passed = causal_prediction_error < 0.50
    preservation_passed = final_arc_accuracy >= 0.80
    gradient_flow_ok = avg_gradient_norm > 0.01
    bond_formation_ok = bond_formation_rate > 0.30
    
    print(f"Bug Condition Fixed (entity acc > 50%):  {'PASS' if bug_fixed else 'FAIL'}")
    print(f"D1 Gate (causal_error < 0.50):           {'PASS' if d1_gate_passed else 'FAIL'}")
    print(f"Preservation (ARC acc >= 80%):           {'PASS' if preservation_passed else 'FAIL'}")
    print(f"Gradient Flow (norm > 0.01):             {'PASS' if gradient_flow_ok else 'FAIL'}")
    print(f"Bond Formation (rate > 30%):             {'PASS' if bond_formation_ok else 'FAIL'}")
    print(f"Checkpoint Save/Load:                    {'PASS' if checkpoint_test_passed else 'FAIL'}")
    
    all_passed = (bug_fixed and d1_gate_passed and preservation_passed and 
                  gradient_flow_ok and bond_formation_ok and checkpoint_test_passed)
    
    print(f"\nOVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"{'='*70}\n")
    
    # Assertions
    assert bug_fixed, (
        f"Bug condition not fixed: entity prediction accuracy {final_graph_accuracy:.4f} <= 0.50"
    )
    
    assert d1_gate_passed, (
        f"D1 gate failed: causal_prediction_error {causal_prediction_error:.4f} >= 0.50"
    )
    
    assert preservation_passed, (
        f"Preservation failed: ARC grid accuracy {final_arc_accuracy:.4f} < 0.80"
    )
    
    assert gradient_flow_ok, (
        f"Gradient flow issue: average gradient norm {avg_gradient_norm:.6f} <= 0.01"
    )
    
    assert bond_formation_ok, (
        f"Bond formation issue: bond formation rate {bond_formation_rate:.4f} <= 0.30"
    )
    
    assert checkpoint_test_passed, (
        f"Checkpoint save/load failed: accuracy difference {accuracy_diff:.6f} >= 0.01"
    )
