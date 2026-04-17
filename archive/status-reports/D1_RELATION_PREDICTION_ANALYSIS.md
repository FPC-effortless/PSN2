# D1 Relation Prediction Accuracy Plateau Analysis

## Executive Summary

The D1 relation prediction accuracy plateaus at **12.8%** despite strong gradient flow (0.45) and 100% bond formation rate. The test expects >50% accuracy but the system is stuck at ~13%. After deep analysis, I've identified **6 critical issues** that explain why the model cannot reach 50% accuracy.

## Test Results (5000 steps)

```
Step  1000 | Loss: 8.9493  | Accuracy: 0.0420 | Grad Norm: 0.169 | Bond Rate: 1.0000
Step  2000 | Loss: 10.0139 | Accuracy: 0.0750 | Grad Norm: 0.433 | Bond Rate: 1.0000
Step  3000 | Loss: 8.1415  | Accuracy: 0.1017 | Grad Norm: 0.576 | Bond Rate: 1.0000
Step  4000 | Loss: 8.0397  | Accuracy: 0.1144 | Grad Norm: 0.647 | Bond Rate: 1.0000
Step  5000 | Loss: 8.8225  | Accuracy: 0.1280 | Grad Norm: 0.710 | Bond Rate: 1.0000

Final: Accuracy 12.8%, Avg Gradient Norm: 0.449, Bond Rate: 100%
```

**Key Observations:**
- ✅ Gradient flow is STRONG (0.45, well above 0.01 threshold)
- ✅ Bond formation is PERFECT (100% of pulses)
- ✅ Steady improvement from 4.2% → 12.8% over 5000 steps
- ❌ Loss remains HIGH (8-10) and doesn't decrease much
- ❌ Accuracy plateaus at ~13%, far below 50% target

## Root Cause Analysis

### Issue 1: **RANDOM GRAPH GENERATION = UNLEARNABLE TASK** ⚠️ CRITICAL

**Location:** `tests/psn2/test_d1_relation_prediction_bug.py:14-30` and `psn2/datasets/rel_graph.py:9-20`

**Problem:** The test batch generation creates **completely random relations** with no semantic structure:

```python
def make_graph_batch(B=4, N_e=6, N_r=5, entity_vocab=64, relation_vocab=32):
    entities = torch.randint(0, entity_vocab, (B, N_e))
    relations = torch.randint(0, relation_vocab, (B, N_r, 3))
    # Relations are RANDOM - no correlation with entities!
    relations[:, :, 0] = torch.randint(0, N_e, (B, N_r))  # random source
    relations[:, :, 2] = torch.randint(0, N_e, (B, N_r))  # random target
```

**Why This Breaks Learning:**
- Entity IDs are random integers (0-63) with NO semantic meaning
- Relations are random triples with NO correlation to entity properties
- The masked entity has NO learnable pattern from its neighbors
- Example: Entity 42 connected to Entity 17 via relation 5 tells you NOTHING about what Entity 42 is

**Analogy:** It's like asking "predict the next lottery number from the previous 5 lottery numbers" - there's no pattern to learn!

**Evidence:**
- Random baseline for 64 classes: 1/64 = 1.56%
- Model achieves 12.8% - only 8x better than random
- Loss stays high (8-10) because the task is fundamentally ambiguous

**What SHOULD Happen:**
- Entities should have semantic properties (e.g., color, shape, size)
- Relations should be deterministic functions of entity properties
- Example: `same_color(red_circle, red_square) = true` is learnable
- Example: `left_of(entity_at_x=2, entity_at_x=5) = true` is learnable

### Issue 2: **ATTENTION MECHANISM USES UNTRAINED EMBEDDINGS** ⚠️ CRITICAL

**Location:** `psn2/core.py:285-290` in `encode_graph_neighborhood`

**Problem:** The attention mechanism uses relation embedding **magnitude** as importance signal, but embeddings are randomly initialized and barely trained:

```python
# Fix 3.1: Attention-weighted aggregation based on relation type
relation_importance = (rel_emb * involved.squeeze(-1).unsqueeze(-1)).norm(dim=-1, keepdim=True)
attention_scores = torch.softmax(relation_importance + (1.0 - involved) * -1e9, dim=1)
```

**Why This Breaks Learning:**
- Early in training, all relation embeddings have similar random magnitudes
- Attention scores are nearly uniform → no discriminative signal
- The attention mechanism doesn't help because it's based on untrained features
- Gradient flow to relation_encoder is weak because attention is saturated

**Evidence:**
- Relation encoder gradient norm is likely much lower than entity encoder
- Attention scores are probably near-uniform (1/N_r) for most samples

**What SHOULD Happen:**
- Use learned attention query/key/value mechanism
- Or use relation type as explicit categorical feature (one-hot)
- Or use multi-head attention to learn multiple relation importance patterns

### Issue 3: **NEIGHBORHOOD CONTEXT LOSES CRITICAL INFORMATION**

**Location:** `psn2/core.py:293-294` in `encode_graph_neighborhood`

**Problem:** The neighborhood aggregation sums over all relations, losing which specific neighbor is connected via which relation:

```python
context = (neighbor_emb + rel_emb) * attention_scores  # [B, N_r, D]
neighborhood = context.sum(dim=1)  # [B, D] - LOSES STRUCTURE!
```

**Why This Breaks Learning:**
- After summing, you can't tell which neighbor had which relation
- For 5 relations, you get a single D-dimensional vector that must encode all neighbor info
- This is a severe information bottleneck
- The model can't learn "if neighbor A has property X and relation is Y, then masked entity has property Z"

**What SHOULD Happen:**
- Use a graph neural network (GNN) layer that preserves neighbor-relation structure
- Or use separate aggregation per relation type
- Or use a transformer that attends to each (neighbor, relation) pair separately

### Issue 4: **MODEL CAPACITY IS INSUFFICIENT FOR 64 CLASSES**

**Location:** `psn2/core.py:145-147` in `PSN2System.__init__`

**Problem:** The model uses dim=128 and max_nodes=32 to predict 64 entity classes from complex relational context:

```python
self.entity_encoder = nn.Embedding(rel_vocab, dim)  # 64 entities → 128-dim
self.entity_decoder = RelationAwareEntityDecoder(dim, rel_vocab, relation_vocab=32)
```

**Why This Is Limiting:**
- 64 entity classes require at least log2(64) = 6 bits of information
- With random entities and relations, the model needs to memorize 64 distinct patterns
- The decoder is a 2-layer MLP: `Linear(256→128) → ReLU → Linear(128→128) → ReLU → Linear(128→64)`
- This is a relatively small network for a 64-way classification task with complex input

**Evidence:**
- Loss stays high (8-10) suggesting the model is capacity-limited
- Cross-entropy loss for random 64-class prediction: -log(1/64) ≈ 4.16
- Observed loss 8-10 is 2x higher, suggesting the model is confused

**What SHOULD Happen:**
- Increase dim to 256 or 512 for richer representations
- Use a deeper decoder (3-4 layers instead of 2)
- Or reduce entity_vocab to 16-32 classes for easier learning

### Issue 5: **DIRECT GRAPH ENCODING BYPASSES NODE BANK** (Design Issue)

**Location:** `psn2/core.py:455-460` in `forward_batch` (graph branch)

**Problem:** Fix 3.2 uses direct graph encoding to bypass the "noisy node bank", but this defeats the purpose of the PSN architecture:

```python
# Fix 3.2: Use direct graph encoding for cleaner gradient path
shape = self.encode_graph(entities, relations)
# Still run pulse cycle for bond formation and other side effects
committed_shape = controller.run_pulse(external_input=shape, regime=phase)
# ...
graph_encoding = shape  # Direct encoding, not filtered through node bank
```

**Why This Is Problematic:**
- The node bank is supposed to learn reusable representations across samples
- By bypassing it, the model can't leverage learned node patterns
- The pulse cycle runs but its output (committed_shape) is ignored for entity prediction
- Bonds are formed between node bank indices, but entity prediction doesn't use them

**What SHOULD Happen:**
- Either fully commit to using the node bank (use committed_shape for prediction)
- Or remove the node bank from graph tasks entirely (don't run pulse cycle)
- The current hybrid approach gets the worst of both worlds

### Issue 6: **LOSS WEIGHTING CREATES CONFLICTING GRADIENTS**

**Location:** `psn2/core.py:505-515` in `forward_batch` (graph branch)

**Problem:** The loss combines entity prediction, relation prediction, shape matching, and spike regularization with competing objectives:

```python
# Fix 3.5: Adjust loss weighting for compositional phase
if phase == "compositional":
    entity_weight = 2.0
    shape_weight = 0.05
else:
    entity_weight = 1.0
    shape_weight = 0.1

loss = entity_weight * loss_entity + 0.5 * loss_relation + shape_weight * loss_shape + 0.01 * loss_spike
```

**Why This Creates Conflicts:**
- `loss_shape` compares active_shape to a re-encoded graph using **predicted** entities
- This creates a circular dependency: entity prediction depends on shape, shape depends on entity prediction
- The shape loss gradient fights against the entity loss gradient
- Even with low weight (0.05), this adds noise to the entity prediction gradients

**Evidence:**
- Loss oscillates (8.9 → 10.0 → 8.1 → 8.0 → 8.8) instead of monotonically decreasing
- This suggests conflicting gradient signals

**What SHOULD Happen:**
- Remove shape loss entirely for graph tasks (set weight to 0.0)
- Or compute shape loss using ground truth entities, not predicted entities
- Focus purely on entity and relation prediction losses

## Why Gradient Flow Is Strong But Accuracy Is Low

**The Paradox:** Gradients are flowing (0.45 norm), but accuracy plateaus at 12.8%.

**Explanation:**
- Gradients measure the **magnitude** of weight updates, not their **usefulness**
- With random data (Issue 1), the model receives strong but **contradictory** gradients
- Example: In sample 1, entity 42 should be predicted from neighbors [17, 23]
- In sample 2, entity 42 should be predicted from neighbors [8, 51]
- There's no consistent pattern, so gradients point in random directions
- The model learns to predict the **average** entity distribution, achieving ~13% accuracy

**Analogy:** It's like trying to navigate with a compass that spins randomly - you're moving (strong gradients) but not toward any destination (low accuracy).

## Why Bond Formation Is 100% But Doesn't Help

**The Paradox:** Bonds form in every pulse, but accuracy doesn't improve.

**Explanation:**
- Bonds are formed between **node bank indices** (0-31), not entity IDs (0-63)
- The mapping from entity IDs to node indices is dynamic and changes every pulse
- Bonds encode relationships between node bank states, not entity relationships
- Entity prediction uses direct graph encoding (Issue 5), bypassing the node bank
- Therefore, bonds are formed but never utilized for entity prediction

**Evidence:**
```python
# Fix 3.4: Strengthen bond utilization during prediction
# Note: Bonds in the current implementation are formed between node bank indices,
# not entity IDs. For graph tasks, we don't have a direct mapping from entity IDs
# to node indices, so bond utilization for entity prediction is limited.
```

## Recommendations to Reach 50% Accuracy

### Priority 1: Fix Data Generation (CRITICAL)

**Create Learnable Synthetic Graphs:**

```python
def make_learnable_graph_batch(B=4, N_e=6, N_r=5):
    """Generate graphs with learnable structure."""
    # Entities have semantic properties: [color, shape, size]
    colors = torch.randint(0, 4, (B, N_e))  # 4 colors
    shapes = torch.randint(0, 4, (B, N_e))  # 4 shapes
    sizes = torch.randint(0, 4, (B, N_e))   # 4 sizes
    
    # Entity ID encodes properties: entity_id = color * 16 + shape * 4 + size
    entities = colors * 16 + shapes * 4 + sizes  # [B, N_e] in range [0, 63]
    
    # Relations are deterministic functions of entity properties
    relations = []
    for b in range(B):
        batch_relations = []
        for i in range(N_e):
            for j in range(i+1, N_e):
                # same_color relation
                if colors[b, i] == colors[b, j]:
                    batch_relations.append([i, 0, j])  # relation type 0 = same_color
                # same_shape relation
                if shapes[b, i] == shapes[b, j]:
                    batch_relations.append([i, 1, j])  # relation type 1 = same_shape
                # larger_than relation
                if sizes[b, i] > sizes[b, j]:
                    batch_relations.append([i, 2, j])  # relation type 2 = larger_than
        
        # Pad to N_r relations
        while len(batch_relations) < N_r:
            batch_relations.append([0, 0, 0])  # padding
        relations.append(batch_relations[:N_r])
    
    relations = torch.tensor(relations, dtype=torch.long)  # [B, N_r, 3]
    
    # Mask a random entity
    masked_entity_idx = torch.randint(0, N_e, (B,))
    target_entity = entities[torch.arange(B), masked_entity_idx]
    
    # Mask token
    entities_masked = entities.clone()
    entities_masked[torch.arange(B), masked_entity_idx] = 63  # mask token
    
    return {
        "type": "graph",
        "entities": entities_masked,
        "relations": relations,
        "target_entity": target_entity,
        "target_relation": torch.randint(0, 3, (B,)),  # dummy for now
        "masked_entity_idx": masked_entity_idx,
    }
```

**Expected Impact:** Accuracy should reach 40-60% with learnable data.

### Priority 2: Simplify Neighborhood Encoding

**Replace attention mechanism with simpler aggregation:**

```python
def encode_graph_neighborhood(self, entities, relations, masked_idx):
    """Simplified neighborhood encoding without attention."""
    B, N_e = entities.shape
    N_r = relations.shape[1]
    
    ent_emb = self.entity_encoder(entities)  # [B, N_e, D]
    rel_type = relations[:, :, 1].clamp(min=0, max=31)
    rel_emb = self.relation_encoder(rel_type)  # [B, N_r, D]
    
    # Find relations involving masked entity
    rel_a = relations[:, :, 0]
    rel_b = relations[:, :, 2]
    m = masked_idx.unsqueeze(1).expand(B, N_r)
    
    is_source = (rel_a == m)
    is_target = (rel_b == m)
    involved = (is_source | is_target).float().unsqueeze(-1)  # [B, N_r, 1]
    
    # Simple mean aggregation of involved relations
    neighbor_context = (rel_emb * involved).sum(dim=1) / (involved.sum(dim=1) + 1e-8)
    
    return normalize(neighbor_context)
```

**Expected Impact:** Reduce complexity, improve gradient flow to relation_encoder.

### Priority 3: Increase Model Capacity

**Increase dim and decoder depth:**

```python
# In PSN2System.__init__
self.dim = 256  # was 128
self.entity_encoder = nn.Embedding(rel_vocab, 256)
self.relation_encoder = nn.Embedding(32, 256)
self.entity_decoder = RelationAwareEntityDecoder(256, rel_vocab, relation_vocab=32)

# In RelationAwareEntityDecoder
self.predict = nn.Sequential(
    nn.Linear(dim, dim),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(dim, dim),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(dim, entity_vocab),
)
```

**Expected Impact:** Better capacity to learn 64-class classification.

### Priority 4: Remove Shape Loss for Graph Tasks

**Eliminate conflicting gradients:**

```python
# In forward_batch, graph branch
if phase == "compositional":
    entity_weight = 2.0
    shape_weight = 0.0  # was 0.05 - REMOVE SHAPE LOSS
else:
    entity_weight = 1.0
    shape_weight = 0.0  # was 0.1 - REMOVE SHAPE LOSS

loss = entity_weight * loss_entity + 0.5 * loss_relation + 0.01 * loss_spike
```

**Expected Impact:** Cleaner gradients for entity prediction.

### Priority 5: Use Committed Shape or Remove Node Bank

**Fix the hybrid approach:**

**Option A: Fully use node bank**
```python
# Use committed_shape for entity prediction
graph_encoding = active_shape  # was: shape (direct encoding)
```

**Option B: Remove node bank for graph tasks**
```python
# Don't run pulse cycle for graph tasks
if btype == "graph":
    shape = self.encode_graph(entities, relations)
    # Skip pulse cycle entirely
    active_shape = shape
    # No bond formation for graph tasks
```

**Expected Impact:** Consistent architecture, better utilization of learned representations.

### Priority 6: Reduce Entity Vocabulary

**Make the task easier:**

```python
# In test and dataset
entity_vocab_size = 16  # was 64
```

**Expected Impact:** 16-class prediction is much easier than 64-class, should reach 50%+ accuracy.

## Conclusion

The D1 relation prediction accuracy plateaus at 12.8% due to **6 fundamental issues**, with the most critical being:

1. **Random graph generation creates an unlearnable task** (Issue 1)
2. **Attention mechanism uses untrained embeddings** (Issue 2)
3. **Neighborhood aggregation loses critical structure** (Issue 3)

The strong gradient flow (0.45) and 100% bond formation rate are **misleading metrics** - they indicate the model is updating weights and forming bonds, but these updates are based on random, contradictory data that has no learnable pattern.

**To reach 50% accuracy, you MUST fix the data generation first (Priority 1).** Without learnable data, no amount of architectural improvements will help. After fixing data generation, apply Priorities 2-6 to improve model capacity and gradient flow.

**Estimated Impact:**
- Priority 1 alone: 12.8% → 35-45% accuracy
- Priorities 1-4 combined: 45-60% accuracy
- All 6 priorities: 60-75% accuracy

The test expectation of >50% accuracy is achievable, but requires fundamental changes to the data generation and model architecture.
