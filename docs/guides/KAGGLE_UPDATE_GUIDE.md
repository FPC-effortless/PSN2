# How to Update Your Kaggle Dataset with Fixes

## Quick Method (Recommended)

### Step 1: Create Updated Zip File

**On Windows:**
```bash
update_kaggle_dataset.bat
```

**On Mac/Linux:**
```bash
chmod +x update_kaggle_dataset.sh
./update_kaggle_dataset.sh
```

This creates `psn2_kaggle_full_v2.zip` with all your fixes.

### Step 2: Upload to Kaggle

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Find your **psn2-kaggle** dataset
3. Click **"New Version"** button
4. Upload `psn2_kaggle_full_v2.zip`
5. Version notes: `Relation prediction fixes - bond formation and masked entity context`
6. Click **"Create"**

### Step 3: Your Notebooks Auto-Update!

- Any existing Kaggle notebooks using `psn2-kaggle` will automatically use the new version
- No need to change anything in your notebook
- Just click "Run All" and it will use the updated code

## Alternative: Manual File Upload

If you prefer to update individual files:

1. Go to your `psn2-kaggle` dataset on Kaggle
2. Click **"New Version"**
3. Click **"Upload Files"**
4. Upload these specific files:
   - `psn2/core.py` (entity decoder + bond usage fixes)
   - `psn2/phases.py` (bond formation fix)
   - `RELATION_PREDICTION_FIX.md` (documentation)
5. Click **"Create"**

## What Changed?

The update includes these critical fixes:

### 1. `psn2/core.py`
- Entity decoder now uses masked position context (2D input instead of 1D)
- Relation decoder uses bond information
- Added bond accuracy loss term

### 2. `psn2/phases.py`
- Bonds now automatically form during compositional/recursive phases
- Forms causal bonds between active nodes

### 3. Documentation
- `RELATION_PREDICTION_FIX.md` explains all changes

## Verify the Update

After uploading, verify in your Kaggle notebook:

```python
# Add this cell to check the version
import psn2.core
import inspect

# Check if entity_decoder has 2D input (dim*2)
model = psn2.core.PSN2System(dim=512, max_nodes=256, grid_vocab=10, rel_vocab=64)
print("Entity decoder input size:", model.entity_decoder.in_features)
# Should print: 1024 (512*2) if updated, 512 if old version

# Check if bond formation code exists
import psn2.phases
source = inspect.getsource(psn2.phases.PhaseController._phase_b)
has_bond_formation = "form_bond" in source
print("Bond formation code present:", has_bond_formation)
# Should print: True if updated
```

## Troubleshooting

### "Dataset not found"
- Make sure you're logged into Kaggle
- Check that your dataset is named exactly `psn2-kaggle`

### "Upload failed"
- File might be too large (max 20GB for free tier)
- The script creates a ~4-5MB zip which is well under the limit

### "Notebook still using old code"
- Click "Edit" on your notebook to create a new version
- The new version will use the latest dataset version
- Or manually change dataset version in notebook settings

### "Permission denied" on script
- On Mac/Linux: run `chmod +x update_kaggle_dataset.sh`
- On Windows: right-click → "Run as Administrator"

## Expected Results After Update

After retraining with the updated code:

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Relation Prediction | 1.5% | 70-85% |
| Entity Prediction | ~1.5% | 40-60% |
| D1 Certification | FAIL | PASS |

You'll need to retrain from scratch or continue training for ~2000-5000 steps to see the improvements.

## Need Help?

- Check `RELATION_PREDICTION_FIX.md` for technical details
- See `KAGGLE_GUIDE.md` for general Kaggle setup
- Open an issue if you encounter problems
