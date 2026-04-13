# Learning Blockers Analysis - Comprehensive Review

## Executive Summary
Completed systematic review of PSN2 codebase to identify issues blocking learning at D1 stage and beyond. Found **8 critical issues** and **5 moderate issues** that need attention.

---

## CRITICAL ISSUES (Blocking D1 Gates)

### 1. ✅ FIXED: Relation Prediction - No Neighborhood Context
**Status**: Fixed in psn2/core.py
**Impact**: D1 gate failure (relation prediction 0.025 vs required >0.80)
**Root Cause**: Entity decoder was using mask token's own embedding instead of relational neighborhood
**Fix Applied**: Added `encode_graph_neighborhood()` method that aggregates neighbor embeddings + relation types

### 2. ✅ FIXED: Grid Accuracy - No Copy Mechanism  
**Status**: Fixed in psn2/core.py
**Impact**: D1 gate failure (grid accuracy 0.807 vs required ≥0.90)
**Root Cause**: Decoder had to reconstruct all cells from scratch, including ~89% unmasked cells
**Fix Applied**: Added residual copy bias (+2.0 * one_hot) in SpatialGridDecoder

### 3. 🔴 CRITICAL: Bond Formation Never Triggers
**Location**: psn2/phases.py, line 103-125
**Impact**: Relational learning completely blocked
**Root Cause**: Bond formation only happens in Phase B when `active_regime in ["compositional", "recursive"]`, but:
- PhaseController.active_regime is initialized to "perceptive" (line 48)
- It's NEVER updated during the pulse cycle
- Training loop sets phase in forward_batch, but PhaseController doesn't see it

**Evidence**:
```python
# psn2/phases.py line 48
self.active_regime = "perceptive"  # Hardcoded, never changes

# psn2/phases.py line 106
if self.active_regime in ["compositional", "recursive"]:
    # This condition NEVER triggers because active_regime stays "perceptive"
```

**Fix Required**:
```python
# In PhaseController.__init__ or run_pulse, accept regime parameter:
def run_pulse(self, external_input, modulatory_input=None, regime="perceptive"):
    self.active_regime = regime  # Update from caller
    # ... rest of pulse cycle
```

### 4. 🔴 CRITICAL: Prune/Merge Runs Too Infrequently
**Location**: psn2/core.py, line 417-420
**Impact**: Node bank fills up, spawn blocked by I-24 budget gate
**Root Cause**: Prune/merge only runs every 500 steps, but spawn attempts every step
**Evidence**: Comment says "Fix #2: Operations 2, 3, 4 — prune/merge every 500 steps (not 10)"
- This is backwards! Running every 10 steps was correct for balance
- At 500-step intervals, nodes accumulate faster than they're freed
- I-24 gate blocks spawn: `spawned <= freed`, but freed only updates every 500 steps

**Fix Required**: Change to every 50-100 steps (not 500):
```python
if global_step % 50 == 0 and global_step > 0:  # Not 500!
    self.growth.maybe_prune_nodes(self.node_bank, global_step)
    self.growth.maybe_merge_nodes(self.node_bank, global_step)
```

### 5. 🔴 CRITICAL: Prune Score Inverted Logic
**Location**: psn2/growth.py, line 127-145
**Impact**: Active nodes get pruned, silent nodes kept
**Root Cause**: Prune score uses LOW error as prune signal, but error is prediction error (high = active learning)
**Evidence**:
```python
# Line 136-137: "Low error → low activity → high prune pressure"
e_val = float(node_bank.e[node_id].item())
activity = min(1.0, e_val / 0.5)   # 0 = silent, 1 = very active
inactivity = 1.0 - activity         # high = silent = prune candidate
```

**Problem**: This is backwards for prediction error!
- High prediction error (e) = node is actively trying to learn = KEEP IT
- Low prediction error (e) = node has converged or is silent = prune candidate is correct

**But**: The code treats e as "activation" when it's actually "prediction error"
- A node with high e is NOT "very active" in a good way
- It's failing to predict, which could mean it's learning OR it's useless

**Fix Required**: Use tau (temporal trace) as primary activity signal, not e:
```python
# tau is the correct activity measure: high tau = recently active
tau_val = float(node_bank.tau[node_id].item())
activity = min(1.0, tau_val / 5.0)  # Use tau, not e
inactivity = 1.0 - activity

# e should be secondary: very high e for long time = stuck node
e_val = float(node_bank.e[node_id].item())
stuck_penalty = 1.0 if e_val > 1.0 else 0.0  # Penalize persistently high error

prune_score = inactivity * age_factor + stuck_penalty
```

### 6. 🔴 CRITICAL: VSA Bind/Unbind Broken for Non-Bipolar Vectors
**Location**: psn2/bonds.py, line 88-95 (form_bond), line 97-110 (recover_source)
**Impact**: Bond recovery fails, relational reasoning blocked
**Root Cause**: Comment says "Fix #3: normalize vectors before bind" but the math is still wrong
**Evidence**:
```python
# Line 91-93: Normalize before bind
src_norm = F.normalize(src_vec.flatten(), dim=0)
tgt_norm = F.normalize(tgt_vec.flatten(), dim=0)
permuted = PermutationIndex.apply(src_norm, type_id)
bond_vec = bind(permuted, tgt_norm)  # bind(a,b) = a*b element-wise

# Line 105-107: Unbind
tgt_norm = F.normalize(tgt_vec.flatten(), dim=0)
unbound = bond.bond_vector * tgt_norm  # Should recover perm(src_norm)
```

**Problem**: For normalized continuous vectors, `bind(a,b) = a*b` and `unbind(c,b) = c*b` does NOT recover `a`!
- Correct unbind for element-wise multiply: `unbind(c, b) = c / b` (not `c * b`)
- But division is unstable for near-zero elements

**Fix Required**: Use XOR-based binding for bipolar vectors OR circular convolution for continuous:
```python
# Option 1: Convert to bipolar before bind
def to_bipolar(vec):
    return torch.sign(vec)

# Option 2: Use circular convolution (proper VSA)
def bind(a, b):
    return torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(b)).real

def unbind(c, b):
    return torch.fft.ifft(torch.fft.fft(c) / (torch.fft.fft(b) + 1e-8)).real
```

### 7. 🟡 MODERATE: Batch Size Must Be Multiple of GPU Count
**Location**: train.py, line 88
**Impact**: Batches silently dropped when batch_size not divisible by n_gpus
**Evidence**:
```python
# Line 88: Force batch_size to be multiple of n_gpus
batch_size = max(n_gpus, (batch_size // n_gpus) * n_gpus)

# Line 273-275: Skip undersized batches
if batch_len < n_gpus:
    continue  # Silently drops data!
```

**Fix Required**: Warn user or adjust batch_size in config validation

### 8. 🟡 MODERATE: Phase Selection Doesn't Match PhaseController
**Location**: train.py, line 280-305
**Impact**: Training uses wrong phase for task type
**Root Cause**: Training loop selects phase based on batch type, but PhaseController ignores it (see Issue #3)
**Evidence**: Phase is passed to forward_batch but never reaches PhaseController.active_regime

---

## MODERATE ISSUES (Performance Degradation)

### 9. 🟡 Local Weight Update Skipped for Batches
**Location**: psn2/phases.py, line 130-145
**Impact**: Local Hebbian learning disabled during training
**Root Cause**: "Fix #8" skips local update for batched input (dim > 1)
**Evidence**:
```python
# Line 138-145
if inp is not None and inp.dim() == 1:
    # Single sample: apply local update
    with torch.no_grad():
        # ... local update code
# For batched input (dim > 1): local update skipped — global AdamW handles it
```

**Problem**: This defeats the dual-mode design (local + global learning)
**Fix**: Apply local update per-sample in batch:
```python
if inp is not None:
    with torch.no_grad():
        if inp.dim() == 1:
            inp = inp.unsqueeze(0)
        for i in range(inp.size(0)):
            # Apply local update for each sample
```

### 10. 🟡 Attractor Cache Not Invalidated on Load
**Location**: psn2/attractor.py, line 54-62
**Impact**: Stale cache after checkpoint load
**Root Cause**: `load_state_dict` doesn't call `_invalidate_cache()`
**Fix**: Add `self._invalidate_cache()` at end of `load_state_dict`

### 11. 🟡 Growth Ledger Doesn't Restore Pending Stability
**Location**: psn2/growth.py, line 283-289
**Impact**: Spawned nodes lose stability tracking across checkpoints
**Evidence**: Comment says "parent_nu tensors not serialized"
**Fix**: Serialize parent_nu or accept that stability windows reset on load

### 12. 🟡 Curiosity Goals Never Aged in Training Loop
**Location**: train.py (missing call)
**Impact**: Stale goals accumulate, saturate 50-goal queue
**Root Cause**: `curiosity.tick_episode()` is called in `maybe_grow()` but only when error > 0.6
**Fix**: Call `tick_episode()` unconditionally every N steps

### 13. 🟡 ERS Promotion Only at Session End
**Location**: psn2/core.py, line 443
**Impact**: Working tier saturates (128 capacity), blocks new memories
**Root Cause**: `ers.attempt_promotions()` called in `maybe_grow()` but not frequently enough
**Fix**: Call `attempt_promotions(session_end=False)` every 100 steps

---

## VERIFICATION PLAN

### Phase 1: Fix Critical Blockers (Issues #3-6)
1. Fix PhaseController.active_regime propagation
2. Adjust prune/merge frequency to 50 steps
3. Fix prune score logic (use tau, not e)
4. Fix VSA bind/unbind (use circular convolution or bipolar)

### Phase 2: Re-run D1 Training
```bash
python train_sequential.py --config configs/default.json \
  --checkpoint-dir /kaggle/working/artifacts \
  --start-stage D1 --end-stage D1
```

### Phase 3: Verify Gates
```bash
python evaluate.py --config configs/default.json \
  --checkpoint /kaggle/working/artifacts/latest.pt
```

**Expected Results**:
- Relation prediction: >0.80 (currently 0.025)
- Grid accuracy: ≥0.90 (currently 0.807)
- Bonds formed: >0 (currently 0)
- Active nodes: stable around 256-512 (not collapsing or exploding)

---

## PRIORITY RANKING

**Must Fix Before Next Training Run**:
1. Issue #3: PhaseController.active_regime propagation (CRITICAL - blocks bond formation)
2. Issue #4: Prune/merge frequency (CRITICAL - blocks node growth)
3. Issue #5: Prune score logic (CRITICAL - kills active nodes)
4. Issue #6: VSA bind/unbind (CRITICAL - blocks relational reasoning)

**Can Fix After Verifying D1 Gates**:
5. Issue #7: Batch size validation (moderate - data loss)
6. Issue #9: Local weight update (moderate - learning efficiency)
7. Issues #10-13: Memory/checkpoint issues (moderate - long-term stability)

---

## NOTES

- Issues #1 and #2 were already fixed in the previous session
- The training did show learning (loss decreased 7.5 → 6.3), so basic gradient flow works
- The gate failures are due to specific architectural issues, not fundamental design flaws
- Once these fixes are applied, the model should pass D1 gates
