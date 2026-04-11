# Sequential Training - Quick Start

## 🚀 Get Started in 3 Steps

### Step 1: Package Your Code
```bash
python update_kaggle_dataset.py
```
Creates `psn2_kaggle_full_v2.zip` with sequential training support.

### Step 2: Upload to Kaggle
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Find `psn2-kaggle` → Click **"New Version"**
3. Upload `psn2_kaggle_full_v2.zip`
4. Version notes: `Sequential training D1→D6 + relation prediction fixes`

### Step 3: Run Sequential Training

**Option A: Kaggle (Recommended)**
1. Upload `kaggle_train_sequential.ipynb` to Kaggle
2. Set GPU to **2×T4**
3. Configure in cell 6:
   ```python
   START_STAGE = 'D1'
   END_STAGE   = 'D6'
   ```
4. Click **"Run All"**

**Option B: Local**
```bash
python train_sequential.py
```

## What Happens

The system will:
1. ✅ Train D1 (20k steps, ~3-4h)
2. ✅ Evaluate D1 and check gates
3. ✅ If gates pass → Train D2 (20k steps, ~3-4h)
4. ✅ Continue through D6
5. ⚠️ Stop if any stage fails gates

## Expected Timeline

| Stage | Steps | Time (2×T4) | Status |
|-------|-------|-------------|--------|
| D1 | 20,000 | ~3-4h | Relation prediction fixed! |
| D2 | 20,000 | ~3-4h | Auto-continues if D1 passes |
| D3 | 15,000 | ~2-3h | Auto-continues if D2 passes |
| D4 | 25,000 | ~4-5h | Auto-continues if D3 passes |
| D5 | 30,000 | ~5-6h | Auto-continues if D4 passes |
| D6 | 40,000 | ~6-8h | Auto-continues if D5 passes |

**Total:** ~24-30 hours (can split across multiple Kaggle sessions)

## Training Strategies

### Strategy 1: One Stage Per Session (Free Tier)
Perfect for Kaggle free tier (30 GPU hours/week):

**Session 1:**
```python
START_STAGE = 'D1'
END_STAGE   = 'D1'
```

**Session 2 (after D1 passes):**
```python
START_STAGE = 'D2'
END_STAGE   = 'D2'
```

Continue through D6.

### Strategy 2: Continuous (Notebooks+)
For Kaggle Notebooks+ ($20/month):

```python
START_STAGE = 'D1'
END_STAGE   = 'D6'
```

Let it run through all stages automatically.

### Strategy 3: Batched (Balanced)
Train 2-3 stages per session:

**Session 1:**
```python
START_STAGE = 'D1'
END_STAGE   = 'D3'
```

**Session 2:**
```python
START_STAGE = 'D4'
END_STAGE   = 'D6'
```

## Monitoring

Watch for stage transitions in the output:
```
======================================================================
STAGE D1
======================================================================
[D1] Starting training...
step=1000 loss=0.2889 nodes=256 attractors=2048 bonds=15
...
[D1] Evaluating...
[D1] Gate Certification:
  ✓ All gates PASSED for D1

======================================================================
STAGE D2
======================================================================
[D2] Starting training...
```

## Results

After completion, check `sequential_results.json`:
```json
{
  "D1": {"passed": true, "eval_results": {...}},
  "D2": {"passed": true, "eval_results": {...}},
  ...
  "D6": {"passed": true, "eval_results": {...}}
}
```

## If Something Fails

### Gates Fail
```python
# Option 1: Continue anyway
SKIP_GATE_CHECK = True

# Option 2: Retrain just that stage
START_STAGE = 'D2'
END_STAGE   = 'D2'
```

### Session Timeout
Checkpoint auto-saves. Next session:
1. Attach `psn2-checkpoint` dataset
2. Set `START_STAGE` to next stage
3. Run all cells

## Files Included

**New in v2:**
- ✅ `train_sequential.py` - Sequential training script
- ✅ `kaggle_train_sequential.ipynb` - Kaggle notebook
- ✅ `SEQUENTIAL_TRAINING_GUIDE.md` - Detailed guide
- ✅ Relation prediction fixes (D1 should pass now!)

**Previous:**
- `train.py` - Single stage training
- `evaluate.py` - Evaluation script
- `psn2/` - Model code
- `configs/` - Configuration files
- `data/` - Training datasets

## What's Fixed

The v2 update includes critical fixes for D1 certification:

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Relation Prediction | 1.5% | 70-85% ✅ |
| Entity Prediction | ~1.5% | 40-60% ✅ |
| D1 Gates | FAIL | PASS ✅ |

## Next Steps

1. ✅ Run `python update_kaggle_dataset.py`
2. ✅ Upload to Kaggle
3. ✅ Start sequential training
4. ✅ Monitor progress
5. ✅ Celebrate when all stages pass! 🎉

## Need Help?

- **Detailed guide:** `SEQUENTIAL_TRAINING_GUIDE.md`
- **Technical fixes:** `RELATION_PREDICTION_FIX.md`
- **Kaggle setup:** `KAGGLE_UPDATE_GUIDE.md`
- **Architecture:** `PRD.md`
