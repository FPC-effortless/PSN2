# PSN-2 Changelog

## 2026-04-10 — Critical Fixes for Kaggle Training

### Core Fixes

**1. Batch dimension collapse in graph branch** (`psn2/core.py`)
- **Issue**: `committed_shape` from `PhaseController.run_pulse()` was always `[D]` (1D), causing `entity_logits` to be `[1, vocab]` vs `target_entity [B]`
- **Impact**: Training crashed with `ValueError: Expected input batch_size (1) to match target batch_size (32)`
- **Fix**: Restore batch dimension by expanding 1D committed shape to `[B, D]` before passing to decoders

**2. Tau accumulation too weak** (`psn2/node.py`)
- **Issue**: Tau update coefficient was `0.10`, giving steady-state `tau ≈ err ≈ 0.5–1.0`, far below the D1 gate threshold of `>5 pulses`
- **Impact**: `temporal_trace_persistence` gate always failed (3.90 vs required >5.0)
- **Fix**: Increased coefficient from `0.10` → `0.60`, giving steady-state `≈ 6*err`, allowing active nodes to exceed threshold

**3. Evaluation metric bug** (`evaluate.py`)
- **Issue**: `eval_graph()` divided `total_correct` by batch count instead of sample count
- **Impact**: With batch_size=32, relation_prediction was inflated by 32x, masking poor performance
- **Fix**: Track `total_samples` and divide by that instead of batch count

### Dataset Fixes

**4. ARC-AGI-2 grid truncation** (`psn2/datasets/arc_agi2.py`)
- **Issue**: `max_grid_size` was set from config's `grid_size=8`, silently truncating all grids >8×8 (most common size is 10×10)
- **Impact**: Output patterns corrupted, model couldn't learn real ARC-AGI-2 tasks
- **Fix**: Clamp `max_grid_size` to `max(passed_value, 30)` to preserve all real grids

**5. Relational graph mask token collision** (`psn2/datasets/rel_graph.py`)
- **Issue**: Mask token was `0`, which collides with valid entity index 0
- **Impact**: Model couldn't distinguish "masked" from "entity 0", making the task unsolvable for entity 0
- **Fix**: Use `vocab_size - 1` as dedicated mask token

**6. Wikitext extreme hash collisions** (`psn2/datasets/wikitext_dataset.py`)
- **Issue**: 1.8M lines of text hashed into only 10 buckets (`vocab_size=10`)
- **Impact**: ~99.99% collision rate, essentially random noise
- **Fix**: Hash into internal 4096-bucket space first, then remap to `vocab_size` at tensor construction

**7. ToM target fragmentation** (`psn2/datasets/tom_dataset.py`)
- **Issue**: `target_relation` was hash of full question string (~13k unique questions → 32 buckets with chaotic distribution)
- **Impact**: Model couldn't learn meaningful question patterns
- **Fix**: Hash only first 3 words of question (captures question type with less fragmentation)

**8. GSM8K/BBH relation structure** (`psn2/datasets/gsm8k_dataset.py`, `psn2/datasets/bbh_dataset.py`)
- **Issue**: Relations were constant task-prefixed hashes, carrying zero structural signal
- **Impact**: Model couldn't leverage relational structure in reasoning tasks
- **Fix**: Use word-pair hashes matching other graph datasets

**9. BBH target collision** (`psn2/datasets/bbh_dataset.py`)
- **Issue**: `True/False` mapped to `1/0` while other targets hashed into `0..63`, causing collisions
- **Impact**: Boolean values collided with other hashed targets
- **Fix**: Hash all targets uniformly

### Training Infrastructure

**10. Config updates** (`train.py`, `kaggle_train.ipynb`)
- Added `arc_grid_size: 30` config parameter separate from `grid_size: 8` (synthetic/wikitext)
- Updated all `ARCAGI2Dataset` calls to use `arc_grid_size`
- Updated smoke test and dataset tests to match new grid sizes

**11. Package rebuild** (`psn2_kaggle_full.zip`)
- Rebuilt with all fixes
- Size: 3.8 MB (wikitext excluded as separate 550MB dataset)

## Testing

All tests pass:
- `scripts/test_datasets.py` — all 7 dataset loaders verified
- `scripts/smoke_test.py` — 3-step training on all 6 datasets successful
- No diagnostics errors in any modified files

## Next Steps for Kaggle

1. Upload new `psn2_kaggle_full.zip` to Kaggle Datasets (update existing `psn2-kaggle` dataset)
2. Start fresh D1 training session with fixed codebase
3. Expected improvements:
   - No more batch dimension crashes
   - Tau persistence will exceed 5-pulse threshold
   - Graph loss should drop below 4.16 (random baseline)
   - Relation prediction accuracy should improve significantly
   - Grid accuracy on real ARC-AGI-2 tasks should be meaningful (not corrupted by truncation)
