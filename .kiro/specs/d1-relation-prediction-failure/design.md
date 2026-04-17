# D1 Relation Prediction Failure Bugfix Design

## Overview

The PSN-2 training system fails the D1 stage gate due to catastrophically low relation prediction accuracy (1.5% vs required 50%+). The system achieves 80%+ accuracy on grid-based tasks but fails completely on relational graph tasks during the compositional phase. This design document analyzes the root causes and proposes targeted fixes to enable relational reasoning without regressing grid performance.

The bug manifests as a gradient flow problem: the entity prediction path through `encode_graph → entity_decoder` fails to learn meaningful representations because (1) the neighborhood context encoding is insufficient, (2) bond formation during compositional phase is not occurring reliably, and (3) the entity decoder receives noisy early-training signals that prevent convergence.

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug - when relational graph batches are processed during compositional phase, entity prediction accuracy remains at random baseline (~1.5%)
- **Property (P)**: The desired behavior - entity prediction accuracy should exceed 50% (error < 0.50) after sufficient training
- **Preservation**: Grid-based task performance (80%+ object tracking) and other D1 gates must remain unchanged
- **encode_graph**: Function in `psn2/core.py` that encodes entities and relations into a VSA shape vector
- **encode_graph_neighborhood**: Function in `psn2/core.py` that builds local context for masked entity prediction
- **entity_decoder**: Neural network in `psn2/core.py` that predicts masked entity from neighborhood + shape context
- **compositional phase**: Training regime where relational bonds are formed between active nodes
- **PhaseController**: Manages A-F pulse cycle and bond formation in `psn2/phases.py`
- **BondSystem**: VSA-based typed bond system in `psn2/bonds.py` using circular convolution

## Bug Details

### Bug Condition

The bug manifests when the system processes relational graph batches during the compositional phase. The `forward_batch` method in `psn2/core.py` encodes the graph, runs a pulse cycle, and attempts to predict a masked entity using neighborhood context. Despite 20,000 training steps, the entity prediction accuracy remains at 1.5% (random baseline for 64-class prediction is 1/64 ≈ 1.56%).

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type dict with keys ["type", "entities", "relations", "target_entity", "masked_entity_idx"]
  OUTPUT: boolean
  
  RETURN input["type"] == "graph"
         AND phase == "compositional"
         AND entity_prediction_accuracy < 0.10
         AND training_steps > 1000
END FUNCTION
```

### Examples

- **Example 1**: Graph with 6 entities, 5 relations, entity at index 2 masked. Expected: predict entity ID from neighborhood context. Actual: random prediction (1.5% accuracy)
- **Example 2**: After 5,000 training steps, relation prediction error = 0.9850 (accuracy 1.5%). Expected: error should be decreasing toward < 0.50. Actual: error remains flat at ~0.98
- **Example 3**: Bond formation during compositional phase should create causal bonds between active nodes. Expected: 2-3 bonds formed per pulse when multiple nodes active. Actual: bonds may not be forming or not being utilized
- **Edge case**: When only 1 node is active, bond formation correctly skips (requires >= 2 nodes). This is expected behavior.

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Grid-based task performance (object tracking accuracy 80%+) must continue to work
- Perceptive phase processing of ARC grids must remain unchanged
- Other D1 gates (temporal_trace_persistence, vsa_binding_accuracy) must continue to pass
- Mixed batch training (60% ARC, 40% graph) must not degrade ARC performance
- DataParallel training across 2 GPUs must continue to function

**Scope:**
All inputs that do NOT involve relational graph batches (type == "graph") should be completely unaffected by this fix. This includes:
- ARC grid batches (type == "arc")
- Grid encoder/decoder paths
- Perceptive phase pulse cycles
- Loss computation for grid tasks

## Hypothesized Root Cause

Based on the bug description and code analysis, the most likely issues are:

1. **Insufficient Neighborhood Context**: The `encode_graph_neighborhood` function aggregates neighbor embeddings but may not provide sufficient discriminative signal for the entity decoder to learn meaningful predictions. The neighborhood context is normalized and averaged, potentially losing critical relational structure.

2. **Bond Formation Not Occurring**: The `PhaseController._phase_b` method forms bonds during compositional phase, but only when `len(active_indices) >= 2`. Early in training, node activation may be sparse, preventing bond formation. Even when bonds form, they are not directly utilized in the entity prediction path.

3. **Noisy Early Training Signal**: The entity decoder receives input from `encode_graph_neighborhood` + `active_shape`. Early in training, `active_shape` is noisy (derived from untrained node bank), and the neighborhood context may be insufficient. This creates a chicken-and-egg problem: the decoder can't learn without good representations, but representations can't improve without decoder gradients.

4. **Gradient Flow Bottleneck**: The gradient path is: `entity_decoder → active_shape → encode_graph → entity_encoder`. If `active_shape` is detached or noisy, gradients don't effectively reach the entity encoder to improve entity embeddings.

5. **Relation Encoder Underutilized**: The relation encoder is used in `encode_graph` but its contribution is averaged with entity embeddings. Relational structure may be diluted in the global shape vector, making it hard for the decoder to extract relational patterns.

## Correctness Properties

Property 1: Bug Condition - Entity Prediction Accuracy

_For any_ relational graph input where the bug condition holds (compositional phase, graph batch, training steps > 1000), the fixed forward_batch function SHALL produce entity predictions with accuracy > 50% (error < 0.50) after sufficient training, significantly better than random baseline (1.5%).

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

Property 2: Preservation - Grid Task Performance

_For any_ input that is NOT a relational graph batch (ARC grid batches, perceptive phase), the fixed code SHALL produce exactly the same behavior as the original code, preserving 80%+ object tracking accuracy and all other D1 gate metrics.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `psn2/core.py`

**Function**: `forward_batch` (graph branch) and `encode_graph_neighborhood`

**Specific Changes**:

1. **Enhance Neighborhood Context Encoding**:
   - Modify `encode_graph_neighborhood` to preserve more relational structure
   - Instead of averaging all neighbor embeddings, use attention-weighted aggregation based on relation type
   - Add relation type embeddings directly to the neighborhood context (not just entity embeddings)
   - Ensure the neighborhood vector is high-dimensional enough to carry discriminative signal

2. **Improve Entity Decoder Input**:
   - Current: `entity_input = torch.cat([neighborhood, active_shape], dim=-1)`
   - Problem: `active_shape` is noisy early in training
   - Fix: Use `encode_graph(entities, relations)` directly instead of `active_shape` for cleaner gradient path
   - This bypasses the noisy node bank and provides direct gradients to entity/relation encoders

3. **Add Relation-Aware Prediction**:
   - The entity decoder should explicitly use relation embeddings from the masked entity's neighborhood
   - Add a relation-conditioned prediction head that takes relation type into account
   - This helps the model learn "entity X is related to entity Y via relation R"

4. **Strengthen Bond Utilization**:
   - Ensure bonds formed in Phase B are actually used during prediction
   - Add a bond-based context retrieval mechanism: query bonds for the masked entity slot
   - Use bond vectors to recover related entities via `BondSystem.recover_source`

5. **Adjust Loss Weighting**:
   - Current: `loss = loss_entity + 0.5 * loss_relation + 0.1 * loss_shape + 0.01 * loss_spike`
   - Problem: Entity loss may be dominated by other terms early in training
   - Fix: Increase entity loss weight to 2.0 during compositional phase to prioritize learning
   - Reduce shape loss weight to 0.05 to prevent interference with entity prediction gradients

6. **Add Gradient Monitoring**:
   - Add gradient norm logging for entity_encoder, relation_encoder, entity_decoder
   - This helps diagnose if gradients are flowing properly during training
   - If gradients are vanishing, may need to adjust learning rate or loss scaling

**File**: `psn2/phases.py`

**Function**: `_phase_b`

**Specific Changes**:

7. **Ensure Bond Formation Reliability**:
   - Current: Forms bonds only when `len(active_indices) >= 2`
   - Problem: Early in training, activation may be sparse
   - Fix: Lower the activation threshold or ensure at least 2 nodes are activated during compositional phase
   - Add logging to track bond formation frequency

8. **Bond Decay Adjustment**:
   - Current: `LAMBDA_BOND = 0.90` (10% decay per pulse)
   - Problem: Bonds may decay too quickly before being utilized
   - Fix: Reduce decay to `LAMBDA_BOND = 0.95` (5% decay) to give bonds more persistence

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that train the model on relational graph batches for 1000 steps and measure entity prediction accuracy. Run these tests on the UNFIXED code to observe failures and understand the root cause. Also inspect gradient norms and bond formation frequency.

**Test Cases**:
1. **Entity Prediction Baseline Test**: Train on 1000 graph batches, measure entity prediction accuracy (will fail on unfixed code - expect ~1.5%)
2. **Gradient Flow Test**: Log gradient norms for entity_encoder, relation_encoder, entity_decoder after 100 steps (will show vanishing or noisy gradients on unfixed code)
3. **Bond Formation Test**: Count bonds formed during compositional phase over 100 pulses (may show zero or very few bonds on unfixed code)
4. **Neighborhood Context Test**: Visualize neighborhood context vectors for masked entities (may show low variance or collapsed representations on unfixed code)

**Expected Counterexamples**:
- Entity prediction accuracy remains at 1.5% after 1000 steps (random baseline)
- Gradient norms for entity_encoder are near zero or highly unstable
- Bond formation occurs in < 10% of pulses due to sparse activation
- Neighborhood context vectors have low cosine variance (< 0.1), indicating insufficient discriminative signal

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := forward_batch_fixed(input, phase="compositional")
  entity_accuracy := compute_accuracy(result["pred"], input["target_entity"])
  ASSERT entity_accuracy > 0.50 after 5000 training steps
  ASSERT gradient_norm(entity_encoder) > 0.01
  ASSERT bond_formation_rate > 0.30
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  result_original := forward_batch_original(input, phase="perceptive")
  result_fixed := forward_batch_fixed(input, phase="perceptive")
  ASSERT result_original["loss"] ≈ result_fixed["loss"] (within 1%)
  ASSERT result_original["pred"].argmax() == result_fixed["pred"].argmax()
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for ARC grid batches, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Grid Task Preservation**: Train on 1000 ARC grid batches, verify object tracking accuracy remains 80%+ after fix
2. **Perceptive Phase Preservation**: Run perceptive phase pulse cycles on grid inputs, verify committed_shape is unchanged
3. **Loss Computation Preservation**: Verify total loss computation for grid batches is unchanged (within 1% tolerance)
4. **DataParallel Preservation**: Verify training on 2 GPUs continues to work without errors

### Unit Tests

- Test `encode_graph_neighborhood` with various graph structures (2 entities, 6 entities, 10 entities)
- Test entity decoder with synthetic neighborhood contexts (verify output shape and gradient flow)
- Test bond formation in Phase B with different activation patterns (0 active, 1 active, 2+ active)
- Test edge cases: empty relations list, all entities masked, single entity graph

### Property-Based Tests

- Generate random graph structures (varying entity count, relation count) and verify entity prediction improves over training
- Generate random entity/relation vocabularies and verify the model doesn't overfit to specific IDs
- Test that bond formation rate increases as training progresses (more nodes become active)
- Test that gradient norms remain stable (not vanishing or exploding) across 1000 training steps

### Integration Tests

- Full training run: 5000 steps on mixed batches (60% ARC, 40% graph), verify D1 gate passes
- Checkpoint save/load: verify entity prediction accuracy is preserved after checkpoint reload
- Multi-GPU training: verify relation prediction accuracy is consistent between single-GPU and 2-GPU training
- Evaluation script: run `evaluate.py` on trained checkpoint, verify relation_prediction > 0.50
