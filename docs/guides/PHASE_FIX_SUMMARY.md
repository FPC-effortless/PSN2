# Phase Selection Fix for D1 Gate Failure

## Problem

D1 training completed 20,000 steps but failed gate certification with:
- ✗ **object_tracking_accuracy**: Grid accuracy 80.7% ✓ but relation prediction 1.5% ✗ (needs ≥90%)
- ✗ **causal_prediction_error**: 98.5% error (needs <20%, i.e., >80% accuracy)
- ✓ **temporal_trace_persistence**: 30.73 pulses ✓
- ✓ **vsa_binding_accuracy**: Passed ✓

## Root Cause

The original training code used **time-based phase progression**:
- First 33% of steps: "perceptive" phase only
- Next 33%: "compositional" phase
- Final 33%: "recursive" phase

However, **graph/relational tasks require "compositional" phase** for proper learning, while **ARC/grid tasks use "perceptive" phase**.

During the first 6,666 steps (33% of 20K), graph batches were processed in "perceptive" phase instead of "compositional" phase, preventing the model from learning relational reasoning.

## Solution

Modified `train.py` to use **batch-type-aware phase selection**:

### New Phase Logic

```python
if batch_type == "graph":
    # Graph tasks (ToM, GSM8K, BBH, relational) → compositional
    phase = "compositional"
    
elif batch_type == "arc":
    # ARC/grid tasks: stage-dependent progression
    if stage in ["D4", "D5", "D6"]:
        # Language stages (Wikitext grids): compositional → recursive
        phase = "recursive" if frac > 0.5 else "compositional"
    else:
        # Spatial reasoning stages: perceptive → compositional
        phase = "compositional" if frac > 0.5 else "perceptive"
```

### Benefits

1. **Graph tasks always use compositional phase** - proper relational reasoning from step 0
2. **ARC tasks use appropriate phases** - perceptive for spatial patterns in D1-D3
3. **Language tasks progress correctly** - compositional → recursive in D4-D6
4. **Stage-aware progression** - different stages emphasize different cognitive abilities

## Dataset Type Mapping

| Dataset | Type Field | Optimal Phase | Used In |
|---------|-----------|---------------|---------|
| ARCGridDataset | `"arc"` | perceptive | D1-D6 |
| ARCAGI2Dataset | `"arc"` | perceptive | D1-D6 |
| RelationalGraphDataset | `"graph"` | compositional | D1-D2 |
| ToMDataset | `"graph"` | compositional | D3 |
| ToMiDataset | `"graph"` | compositional | D3 |
| WikitextDataset | `"arc"` | recursive | D4 |
| GSM8KDataset | `"graph"` | compositional | D5 |
| BBHDataset | `"graph"` | compositional | D6 |

## Expected Impact

With this fix, D1 training should achieve:
- **Relation prediction**: >80% (was 1.5%)
- **Causal prediction error**: <20% (was 98.5%)
- **Gate certification**: All D1 gates should pass

## Next Steps

1. **Retrain D1 from scratch** with the fixed phase logic:
   ```bash
   python train_sequential.py --config configs/default.json \
     --checkpoint-dir /kaggle/working/artifacts \
     --start-stage D1 --end-stage D6
   ```

2. **Monitor relation prediction** during training - should improve steadily

3. **Verify gate passage** - D1 should certify and progress to D2

## Files Modified

- `train.py`: Updated phase selection logic (lines ~263-290)
