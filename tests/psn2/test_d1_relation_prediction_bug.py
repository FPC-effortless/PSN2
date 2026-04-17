"""Bug condition exploration test for D1 relation prediction failure.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5**

This test encodes the EXPECTED behavior for relation prediction during compositional phase.
It is designed to FAIL on unfixed code (confirming the bug exists) and PASS after the fix.

The test verifies:
- Entity prediction accuracy > 50% after sufficient training
- Gradient norms for entity_encoder > 0.01 (not vanishing)
- Bond formation rate > 30% of pulses
"""
import torch
import pytest
from hypothesis import given, strategies as st, settings, Phase
from psn2.core import PSN2System
from psn2.datasets import RelationalGraphDataset


def make_graph_batch(B=4, N_e=6, N_r=5, entity_vocab=64, relation_vocab=32):
    """Create a relational graph batch with LEARNABLE semantic structure.

    Simplified to 4 entities (2 colors × 2 shapes) so the oracle accuracy
    exceeds 50% (empirically ~66%) and the task is achievable.

    Entity ID encoding: entity_id = color * 2 + shape
      - 0 = red-circle, 1 = red-square, 2 = blue-circle, 3 = blue-square
    Mask token: 4 (out of vocab range 0-3)

    Relations are deterministic:
      - Relation type 0: same_color  (constrains color)
      - Relation type 1: same_shape  (constrains shape)

    With both relation types present, the masked entity is uniquely determined.
    Oracle accuracy ≈ 66% (some samples have only one relation type available).
    """
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
        # so the task is fully determined. Fall back to any entity with a relation.
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


@pytest.mark.slow
def test_bug_condition_relation_prediction_catastrophic_failure():
    """
    **Property 1: Bug Condition** - Relation Prediction Catastrophic Failure
    
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5**
    
    **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists.
    **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bug exists)
    
    This test encodes the expected behavior:
    - Entity prediction accuracy > 50% (error < 0.50)
    - Gradient norms for entity_encoder > 0.01 (not vanishing)
    - Bond formation rate > 30% of pulses
    
    When this test passes after implementing the fix, it confirms the bug is resolved.
    """
    # Initialize model with 4-entity vocabulary (2 colors × 2 shapes + 1 mask token)
    # Oracle accuracy for this setup is ~66%, well above the 50% target.
    torch.manual_seed(42)
    model = PSN2System(dim=256, max_nodes=32, grid_vocab=10, rel_vocab=5, stage="D1")
    model.train()

    # Use Adam optimizer for stable training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training configuration
    num_steps = 5000  # 5000 steps is sufficient for 4-entity task
    batch_size = 4
    num_entities = 6
    num_relations = 5
    entity_vocab_size = 5   # 4 entities + 1 mask token
    relation_vocab_size = 32
    
    # Metrics tracking
    correct_predictions = 0
    total_predictions = 0
    gradient_norms = []
    bond_formation_count = 0
    total_pulses = 0
    
    print(f"\n{'='*70}")
    print(f"Bug Condition Exploration Test - Training for {num_steps} steps")
    print(f"{'='*70}")
    
    # Training loop
    for step in range(num_steps):
        # Generate batch
        batch = make_graph_batch(B=batch_size, N_e=num_entities, N_r=num_relations, 
                                entity_vocab=entity_vocab_size, relation_vocab=relation_vocab_size)
        
        # Forward pass
        optimizer.zero_grad()
        output = model.forward_batch(batch, phase="compositional")
        loss = output["loss"]
        
        # Backward pass
        loss.backward()
        
        # Track gradient norms for entity_encoder
        if hasattr(model, 'entity_encoder'):
            # Entity encoder is now PropertyAwareEntityEncoder with multiple parameters
            entity_encoder_grad_norm = 0.0
            entity_encoder_param_count = 0
            for param in model.entity_encoder.parameters():
                if param.grad is not None:
                    entity_encoder_grad_norm += param.grad.norm().item() ** 2
                    entity_encoder_param_count += 1
            if entity_encoder_param_count > 0:
                grad_norm = (entity_encoder_grad_norm / entity_encoder_param_count) ** 0.5
                gradient_norms.append(grad_norm)
        
        # Track bond formation using model's bond formation stats
        if hasattr(model, 'get_bond_formation_stats'):
            stats = model.get_bond_formation_stats()
            bond_formation_count = stats['bond_formation_count']
            total_pulses = stats['total_pulses']
        
        # Update weights
        optimizer.step()
        
        # Track accuracy
        with torch.no_grad():
            pred_entities = output["pred"].argmax(dim=-1)
            target_entities = batch["target_entity"]
            correct = (pred_entities == target_entities).sum().item()
            correct_predictions += correct
            total_predictions += batch_size
        
        # Log progress every 2000 steps
        if (step + 1) % 2000 == 0:
            current_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            avg_grad_norm = sum(gradient_norms[-100:]) / len(gradient_norms[-100:]) if gradient_norms else 0.0
            bond_rate = bond_formation_count / total_pulses if total_pulses > 0 else 0.0
            
            print(f"Step {step + 1:5d} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Accuracy: {current_accuracy:.4f} | "
                  f"Grad Norm: {avg_grad_norm:.6f} | "
                  f"Bond Rate: {bond_rate:.4f}")
    
    # Final metrics
    final_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    avg_gradient_norm = sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0
    bond_formation_rate = bond_formation_count / total_pulses if total_pulses > 0 else 0.0
    
    print(f"\n{'='*70}")
    print(f"Final Metrics After {num_steps} Steps:")
    print(f"{'='*70}")
    print(f"Entity Prediction Accuracy: {final_accuracy:.4f} (target: > 0.50)")
    print(f"Average Gradient Norm:      {avg_gradient_norm:.6f} (target: > 0.01)")
    print(f"Bond Formation Rate:        {bond_formation_rate:.4f} (target: > 0.30)")
    print(f"{'='*70}")
    
    # Document counterexamples (expected on unfixed code)
    if final_accuracy <= 0.50:
        print(f"\n! COUNTEREXAMPLE FOUND (Expected on unfixed code):")
        print(f"   Entity prediction accuracy: {final_accuracy:.4f} (<= 0.50)")
        print(f"   This confirms the bug exists: accuracy remains below target")
    
    if avg_gradient_norm <= 0.01:
        print(f"\n! COUNTEREXAMPLE FOUND (Expected on unfixed code):")
        print(f"   Gradient norm: {avg_gradient_norm:.6f} (<= 0.01)")
        print(f"   This confirms vanishing gradients in entity_encoder")
    
    if bond_formation_rate <= 0.30:
        print(f"\n! COUNTEREXAMPLE FOUND (Expected on unfixed code):")
        print(f"   Bond formation rate: {bond_formation_rate:.4f} (<= 0.30)")
        print(f"   This confirms insufficient bond formation during compositional phase")
    
    print(f"\n{'='*70}")
    print(f"Test Status: This test encodes EXPECTED behavior")
    print(f"On UNFIXED code: Test should FAIL (confirming bug exists)")
    print(f"After fix: Test should PASS (confirming bug is resolved)")
    print(f"{'='*70}\n")
    
    # Assertions - these encode the EXPECTED behavior
    # On unfixed code, these will FAIL (which is correct - it proves the bug)
    # After the fix, these will PASS (confirming the bug is resolved)
    assert final_accuracy > 0.50, (
        f"Entity prediction accuracy {final_accuracy:.4f} ≤ 0.50. "
        f"Expected: > 50% accuracy after {num_steps} training steps. "
        f"This failure confirms the relation prediction bug exists."
    )
    
    assert avg_gradient_norm > 0.01, (
        f"Average gradient norm {avg_gradient_norm:.6f} ≤ 0.01. "
        f"Expected: gradient norms > 0.01 (not vanishing). "
        f"This failure confirms gradients are not flowing properly to entity_encoder."
    )
    
    assert bond_formation_rate > 0.30, (
        f"Bond formation rate {bond_formation_rate:.4f} ≤ 0.30. "
        f"Expected: bond formation rate > 30% of pulses. "
        f"This failure confirms bonds are not forming reliably during compositional phase."
    )


@pytest.mark.slow
@given(
    num_entities=st.integers(min_value=4, max_value=8),
    num_relations=st.integers(min_value=3, max_value=7),
)
@settings(
    max_examples=3,  # Limited examples for scoped PBT approach
    phases=[Phase.generate, Phase.target],
    deadline=None,  # No deadline for long-running training
)
def test_bug_condition_property_based(num_entities, num_relations):
    """
    **Property 1: Bug Condition (Property-Based)** - Relation Prediction Failure
    
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5**
    
    Property-based test that generates different graph structures to explore
    the bug condition across various input configurations.
    
    **EXPECTED OUTCOME**: Test FAILS on unfixed code (confirms bug exists)
    """
    torch.manual_seed(42)
    model = PSN2System(dim=256, max_nodes=32, grid_vocab=10, rel_vocab=5, stage="D1")
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Shorter training for property-based test (multiple examples)
    num_steps = 1000
    batch_size = 4
    entity_vocab_size = 5   # 4 entities + 1 mask token
    relation_vocab_size = 32

    correct_predictions = 0
    total_predictions = 0

    for step in range(num_steps):
        batch = make_graph_batch(B=batch_size, N_e=num_entities, N_r=num_relations,
                                entity_vocab=entity_vocab_size, relation_vocab=relation_vocab_size)

        optimizer.zero_grad()
        output = model.forward_batch(batch, phase="compositional")
        loss = output["loss"]
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_entities = output["pred"].argmax(dim=-1)
            target_entities = batch["target_entity"]
            correct = (pred_entities == target_entities).sum().item()
            correct_predictions += correct
            total_predictions += batch_size

    final_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    # For property-based test, we expect accuracy to improve beyond random baseline
    # Random baseline for 4 classes is 25%
    # After 1000 steps, we should see at least 40% accuracy if learning is working
    assert final_accuracy > 0.40, (
        f"Entity prediction accuracy {final_accuracy:.4f} ≤ 0.40 after {num_steps} steps "
        f"with {num_entities} entities and {num_relations} relations. "
        f"Expected: > 40% accuracy (significantly better than random baseline ~25%). "
        f"This failure confirms the relation prediction bug exists."
    )
