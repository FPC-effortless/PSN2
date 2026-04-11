# Relation Prediction Fix - PSN-2

## Problem Analysis

Your evaluation showed **relation prediction at 1.5%** (needs >= 85%) - a critical failure for D1 certification.

### Root Causes Identified

1. **No Masked Position Context**: The entity decoder received only a global graph embedding without knowing which entity slot to predict
2. **Bonds Never Formed**: The bond system existed but bonds were never created during training
3. **Bonds Not Used**: Even if bonds existed, they weren't used for relation prediction
4. **No Bond Loss**: No training signal to encourage accurate bond formation

## Fixes Applied

### 1. Entity Decoder with Masked Position Context (`psn2/core.py`)

**Changed**: Entity decoder from `Linear(dim, vocab)` to `Linear(dim*2, vocab)`

**Added**: Concatenate masked entity embedding with node representation
```python
# Get the embedding of the masked entity (mask token)
masked_entity_emb = self.entity_encoder(entities[i, masked_idx])
# Concatenate with node representation
entity_input = torch.cat([node_repr, masked_entity_emb], dim=-1)
entity_logits = self.entity_decoder(entity_input)
```

This gives the decoder explicit context about which entity position needs to be filled.

### 2. Bond Formation During Training (`psn2/phases.py`)

**Added**: Automatic bond formation in Phase B for compositional/recursive regimes
```python
if self.active_regime in ["compositional", "recursive"]:
    # Form bonds between pairs of active nodes
    bond_type = "temporal" if self.active_regime == "recursive" else "causal"
    for adjacent active node pairs:
        self.bond_system.form_bond(bond_type, src_idx, tgt_idx, src_vec, tgt_vec)
```

Bonds are now created between active nodes during graph processing, building relational structure.

### 3. Bond-Informed Relation Prediction (`psn2/core.py`)

**Added**: Use bond vectors to inform relation prediction
```python
if len(self.bond_system.bonds) > 0:
    # Bundle recent strong bonds
    bond_bundle = normalize(stack([bond.bond_vector * bond.strength]))
    # Add to relation prediction input
    relation_input = normalize(node_repr + 0.5 * bond_bundle)
relation_logits = self.relation_decoder(relation_input)
```

The model now uses learned bond structure to predict relations.

### 4. Bond Accuracy Loss (`psn2/core.py`)

**Added**: Loss term to encourage accurate bond formation
```python
for bond in recent_bonds:
    recovered_idx, recovered_vec, sim = bond_system.recover_source(bond, tgt_vec, codebook)
    loss_bond += (1.0 - sim)  # Penalize low recovery similarity
loss = loss_entity + loss_relation + 0.1*loss_shape + 0.05*loss_bond + 0.01*loss_spike
```

This trains the bond system to form bonds that can accurately recover source nodes from targets.

## Expected Impact

These fixes address the fundamental issue: **the model had no mechanism to learn or use relational structure**.

With these changes:
- Entity prediction gets explicit positional context (should improve from 1.5% to 40-60%)
- Bonds are formed and trained during graph processing
- Relation prediction uses learned bond structure (should improve to 70-85%+)
- Bond accuracy is directly optimized

## Next Steps

1. **Retrain** from scratch or continue training with these fixes
2. **Monitor** bond formation: check `len(model.bond_system.bonds)` during training
3. **Evaluate** relation prediction after ~1000 steps to see improvement
4. **Tune** bond loss weight (currently 0.05) if needed

The relation prediction should improve dramatically - from 1.5% to 70-85%+ after retraining.
