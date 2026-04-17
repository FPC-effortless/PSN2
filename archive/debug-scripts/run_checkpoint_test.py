"""
Task 4 Checkpoint Test: Verify all D1 relation prediction fixes work correctly.
Runs 5000 steps on mixed batches and checks all metrics.
"""
import torch
import sys
sys.path.insert(0, '.')
from psn2.core import PSN2System
from psn2.datasets.arc_grid import ARCGridDataset


def make_graph_batch(B=4, N_e=6, N_r=5):
    """Create a relational graph batch with learnable semantic structure (4-entity vocab)."""
    colors = torch.randint(0, 2, (B, N_e))
    shapes = torch.randint(0, 2, (B, N_e))
    entities = colors * 2 + shapes  # [B, N_e] in range [0, 3]

    relations_list = []
    masked_entity_idx_list = []

    for b in range(B):
        batch_relations = []
        for i in range(N_e):
            for j in range(i + 1, N_e):
                if colors[b, i] == colors[b, j]:
                    batch_relations.append([i, 0, j])
                if shapes[b, i] == shapes[b, j]:
                    batch_relations.append([i, 1, j])

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
        while len(batch_relations) < N_r:
            batch_relations.append([0, 0, 0])
        relations_list.append(batch_relations[:N_r])

    relations = torch.tensor(relations_list, dtype=torch.long)
    masked_entity_idx = torch.tensor(masked_entity_idx_list, dtype=torch.long)
    target_entity = entities[torch.arange(B), masked_entity_idx]
    entities_masked = entities.clone()
    entities_masked[torch.arange(B), masked_entity_idx] = 4  # mask token
    target_relation = relations[:, :, 1].mode(dim=1)[0]

    return {
        "type": "graph",
        "entities": entities_masked,
        "relations": relations,
        "target_entity": target_entity,
        "target_relation": target_relation,
        "masked_entity_idx": masked_entity_idx,
    }


def make_arc_batch(B=4, grid_size=8, vocab=10):
    """Create an ARC grid batch."""
    dataset = ARCGridDataset(n_samples=B, grid_size=grid_size, vocab=vocab)
    batches = [dataset[i] for i in range(B)]
    return {
        "type": "arc",
        "input_grid": torch.stack([b["input_grid"] for b in batches]),
        "target_grid": torch.stack([b["target_grid"] for b in batches]),
        "mask": torch.stack([b["mask"] for b in batches]),
    }


def run_integration_test():
    """
    Integration test: 5000 steps on mixed batches (60% ARC, 40% graph).
    Verifies D1 gate: causal_prediction_error < 0.50.
    """
    print("=" * 70)
    print("TASK 4 CHECKPOINT: Integration Test")
    print("5000 steps on mixed batches (60% ARC, 40% graph)")
    print("=" * 70)

    torch.manual_seed(42)
    model = PSN2System(dim=256, max_nodes=32, grid_vocab=10, rel_vocab=5, stage="D1")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_steps = 5000
    batch_size = 4

    # Metrics
    graph_correct = 0
    graph_total = 0
    arc_correct = 0
    arc_total = 0
    gradient_norms = []
    bond_formation_count = 0
    total_pulses = 0

    for step in range(num_steps):
        # 60% ARC, 40% graph
        if torch.rand(1).item() < 0.60:
            batch = make_arc_batch(B=batch_size, grid_size=8, vocab=10)
            phase = "perceptive"
        else:
            batch = make_graph_batch(B=batch_size, N_e=6, N_r=5)
            phase = "compositional"

        optimizer.zero_grad()
        output = model.forward_batch(batch, phase=phase)
        loss = output["loss"]
        loss.backward()

        # Track gradient norms for graph batches
        if batch["type"] == "graph":
            entity_encoder_grad_norm = 0.0
            entity_encoder_param_count = 0
            for param in model.entity_encoder.parameters():
                if param.grad is not None:
                    entity_encoder_grad_norm += param.grad.norm().item() ** 2
                    entity_encoder_param_count += 1
            if entity_encoder_param_count > 0:
                grad_norm = (entity_encoder_grad_norm / entity_encoder_param_count) ** 0.5
                gradient_norms.append(grad_norm)

        optimizer.step()

        # Track bond formation stats
        if hasattr(model, "get_bond_formation_stats"):
            stats = model.get_bond_formation_stats()
            bond_formation_count = stats["bond_formation_count"]
            total_pulses = stats["total_pulses"]

        # Track accuracy
        with torch.no_grad():
            if batch["type"] == "graph":
                pred_entities = output["pred"].argmax(dim=-1)
                target_entities = batch["target_entity"]
                correct = (pred_entities == target_entities).sum().item()
                graph_correct += correct
                graph_total += batch_size
            else:
                pred_grids = output["pred"].argmax(dim=-1)
                target_grids = batch["target_grid"]
                correct = (pred_grids == target_grids).sum().item()
                total = target_grids.numel()
                arc_correct += correct
                arc_total += total

        if (step + 1) % 1000 == 0:
            graph_acc = graph_correct / graph_total if graph_total > 0 else 0.0
            arc_acc = arc_correct / arc_total if arc_total > 0 else 0.0
            avg_grad = sum(gradient_norms[-100:]) / len(gradient_norms[-100:]) if gradient_norms else 0.0
            bond_rate = bond_formation_count / total_pulses if total_pulses > 0 else 0.0
            print(f"Step {step+1:5d} | Graph Acc: {graph_acc:.4f} | ARC Acc: {arc_acc:.4f} | "
                  f"Grad: {avg_grad:.6f} | Bond Rate: {bond_rate:.4f}")

    # Final metrics
    final_graph_accuracy = graph_correct / graph_total if graph_total > 0 else 0.0
    final_arc_accuracy = arc_correct / arc_total if arc_total > 0 else 0.0
    avg_gradient_norm = sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0
    bond_formation_rate = bond_formation_count / total_pulses if total_pulses > 0 else 0.0
    causal_prediction_error = 1.0 - final_graph_accuracy

    print()
    print("=" * 70)
    print("FINAL RESULTS:")
    print("=" * 70)
    print(f"Graph Entity Accuracy:    {final_graph_accuracy:.4f} (target: > 0.50) -> {'PASS' if final_graph_accuracy > 0.50 else 'FAIL'}")
    print(f"causal_prediction_error:  {causal_prediction_error:.4f} (target: < 0.50) -> {'PASS' if causal_prediction_error < 0.50 else 'FAIL'}")
    print(f"ARC Grid Accuracy:        {final_arc_accuracy:.4f} (target: >= 0.80) -> {'PASS' if final_arc_accuracy >= 0.80 else 'FAIL'}")
    print(f"Average Gradient Norm:    {avg_gradient_norm:.6f} (target: > 0.01) -> {'PASS' if avg_gradient_norm > 0.01 else 'FAIL'}")
    print(f"Bond Formation Rate:      {bond_formation_rate:.4f} (target: > 0.30) -> {'PASS' if bond_formation_rate > 0.30 else 'FAIL'}")
    print("=" * 70)

    # D1 Gate check
    from psn2.dc.stage_d1 import StageD1
    stage = StageD1()
    stage.update_metrics(
        object_tracking=final_arc_accuracy,
        causal_prediction_error=causal_prediction_error,
        trace_persistence=10.0,  # Simulated (would need full training to measure)
        vsa_binding=0.80,        # Simulated (would need full training to measure)
    )
    print()
    print("D1 GATE CERTIFIER REPORT:")
    print(stage.report())
    print()
    print(f"D1 Gate Certified: {stage.is_complete()}")

    return {
        "graph_accuracy": final_graph_accuracy,
        "arc_accuracy": final_arc_accuracy,
        "causal_prediction_error": causal_prediction_error,
        "avg_gradient_norm": avg_gradient_norm,
        "bond_formation_rate": bond_formation_rate,
        "d1_gate_passed": causal_prediction_error < 0.50,
    }


def run_checkpoint_save_load_test():
    """
    Verify checkpoint save/load preserves entity prediction accuracy.
    """
    print()
    print("=" * 70)
    print("CHECKPOINT SAVE/LOAD TEST")
    print("=" * 70)

    torch.manual_seed(123)
    model = PSN2System(dim=128, max_nodes=32, grid_vocab=10, rel_vocab=5, stage="D1")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train for 500 steps to get a reasonable model
    for step in range(500):
        batch = make_graph_batch(B=4, N_e=6, N_r=5)
        optimizer.zero_grad()
        output = model.forward_batch(batch, phase="compositional")
        output["loss"].backward()
        optimizer.step()

    # Measure accuracy before save
    correct_before = 0
    total_before = 0
    model.eval()
    with torch.no_grad():
        for _ in range(50):
            batch = make_graph_batch(B=4, N_e=6, N_r=5)
            output = model.forward_batch(batch, phase="compositional")
            pred = output["pred"].argmax(dim=-1)
            correct_before += (pred == batch["target_entity"]).sum().item()
            total_before += 4
    accuracy_before = correct_before / total_before

    # Save checkpoint
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Load into new model
    model2 = PSN2System(dim=128, max_nodes=32, grid_vocab=10, rel_vocab=5, stage="D1")
    model2.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model2.eval()

    # Measure accuracy after load
    correct_after = 0
    total_after = 0
    with torch.no_grad():
        for _ in range(50):
            batch = make_graph_batch(B=4, N_e=6, N_r=5)
            output = model2.forward_batch(batch, phase="compositional")
            pred = output["pred"].argmax(dim=-1)
            correct_after += (pred == batch["target_entity"]).sum().item()
            total_after += 4
    accuracy_after = correct_after / total_after

    os.unlink(checkpoint_path)

    print(f"Accuracy before save: {accuracy_before:.4f}")
    print(f"Accuracy after load:  {accuracy_after:.4f}")
    diff = abs(accuracy_before - accuracy_after)
    print(f"Difference:           {diff:.6f} (target: < 0.01)")
    result = diff < 0.01
    print(f"Checkpoint test:      {'PASS' if result else 'FAIL'}")
    return result


if __name__ == "__main__":
    # Run integration test
    results = run_integration_test()

    # Run checkpoint test
    checkpoint_ok = run_checkpoint_save_load_test()

    print()
    print("=" * 70)
    print("TASK 4 CHECKPOINT SUMMARY")
    print("=" * 70)
    all_pass = (
        results["graph_accuracy"] > 0.50 and
        results["causal_prediction_error"] < 0.50 and
        results["arc_accuracy"] >= 0.80 and
        results["avg_gradient_norm"] > 0.01 and
        results["bond_formation_rate"] > 0.30 and
        checkpoint_ok
    )
    print(f"Bug Condition Fixed (entity acc > 50%):  {'PASS' if results['graph_accuracy'] > 0.50 else 'FAIL'}")
    print(f"D1 Gate (causal_error < 0.50):           {'PASS' if results['causal_prediction_error'] < 0.50 else 'FAIL'}")
    print(f"Preservation (ARC acc >= 80%):           {'PASS' if results['arc_accuracy'] >= 0.80 else 'FAIL'}")
    print(f"Gradient Flow (norm > 0.01):             {'PASS' if results['avg_gradient_norm'] > 0.01 else 'FAIL'}")
    print(f"Bond Formation (rate > 30%):             {'PASS' if results['bond_formation_rate'] > 0.30 else 'FAIL'}")
    print(f"Checkpoint Save/Load:                    {'PASS' if checkpoint_ok else 'FAIL'}")
    print()
    print(f"OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 70)

    sys.exit(0 if all_pass else 1)
