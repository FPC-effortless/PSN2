# Retraining Guide After Phase Fix

## What Was Fixed

The training phase selection logic now correctly assigns:
- **Graph batches** → `compositional` phase (was incorrectly using `perceptive` in first 33% of training)
- **ARC batches** → `perceptive` → `compositional` progression for D1-D3
- **Language batches** → `compositional` → `recursive` progression for D4-D6

This fix should resolve the D1 gate failure where relation prediction was only 1.5% (needed >80%).

## Option 1: Retrain D1 from Scratch (Recommended)

Start fresh with the corrected phase logic:

```bash
# Remove old checkpoint to start clean
rm /kaggle/working/artifacts/latest.pt

# Train D1 → D6 with fixed phase logic
python train_sequential.py \
  --config configs/default.json \
  --checkpoint-dir /kaggle/working/artifacts \
  --start-stage D1 \
  --end-stage D6
```

**Expected results:**
- D1 relation prediction: >80% (was 1.5%)
- D1 causal prediction error: <20% (was 98.5%)
- All D1 gates should pass
- Training will automatically continue to D2-D6

## Option 2: Continue from D1 Checkpoint (Faster but Less Optimal)

If you want to continue from the existing checkpoint and just retrain D1:

```bash
# Retrain D1 with the fixed logic
python train_sequential.py \
  --config configs/default.json \
  --checkpoint-dir /kaggle/working/artifacts \
  --start-stage D1 \
  --end-stage D1 \
  --resume /kaggle/working/artifacts/latest.pt
```

**Note:** This will continue from step 20,000, so the model already has some incorrect learning. Option 1 is better.

## Option 3: Skip Gates and Continue (Not Recommended)

If you want to see what happens with the current checkpoint:

```bash
python train_sequential.py \
  --config configs/default.json \
  --checkpoint-dir /kaggle/working/artifacts \
  --start-stage D2 \
  --end-stage D6 \
  --resume /kaggle/working/artifacts/latest.pt \
  --skip-gate-check
```

**Warning:** D2-D6 may also fail if D1 didn't learn relational reasoning properly.

## Monitoring Training

Watch for these metrics during D1 training:

```
step=X stage=D1 phase=compositional loss=Y.YYYY pred=Z.ZZZZ
```

- **Graph batches** should show `phase=compositional` from step 0
- **Loss should decrease** steadily for both ARC and graph batches
- **Relation prediction** in evaluation should be >80%

## Verification After Training

After D1 completes, check the evaluation output:

```bash
cat /kaggle/working/artifacts/eval_D1.json
```

Look for:
```json
{
  "grid_accuracy": 0.80+,
  "relation_prediction": 0.80+,  // Should be >0.80 now (was 0.015)
  "d1_gates": {
    "object_tracking_accuracy": true,
    "causal_prediction_error": true,
    "temporal_trace_persistence": true,
    "vsa_binding_accuracy": true
  }
}
```

## Estimated Training Time

- **D1**: ~2 hours (20,000 steps)
- **D2**: ~2 hours (20,000 steps)
- **D3**: ~1.5 hours (15,000 steps)
- **D4**: ~2.5 hours (25,000 steps)
- **D5**: ~3 hours (30,000 steps)
- **D6**: ~4 hours (40,000 steps)

**Total**: ~15 hours for full D1-D6 training

## Troubleshooting

### If D1 still fails gates:

1. **Check relation prediction**: Should be >80%, not 1.5%
2. **Verify phase usage**: Look for `phase=compositional` in graph batch logs
3. **Increase D1 steps**: Edit `train_sequential.py` to give D1 more training (e.g., 30,000 steps)
4. **Check data mix**: Ensure 40% of batches are graph batches

### If training is too slow:

1. **Reduce batch size**: Edit `configs/default.json` to lower `batch_size`
2. **Use fewer samples**: Set `max_arc2_samples` in config
3. **Train fewer stages**: Use `--end-stage D3` to stop earlier

## Files Modified

- `train.py`: Phase selection logic (lines ~263-290)
- `PHASE_FIX_SUMMARY.md`: Detailed explanation of the fix
- `test_phase_selection.py`: Test script to verify phase logic
