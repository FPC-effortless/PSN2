# PSN-2 Kaggle Training Guide

## Quick Start (First Session)

### 1. Upload Dataset
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click **New Dataset**
3. Upload `psn2_kaggle_full.zip` (3.8 MB)
4. Name it `psn2-kaggle`
5. Kaggle will auto-unzip it

### 2. Setup Kaggle API Credentials
1. Go to **Account → Settings → API → Create New Token**
2. Download `kaggle.json`
3. Extract `username` and `key` from the JSON

### 3. Create Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **New Notebook**
3. Settings:
   - **Accelerator**: GPU T4 x2 (or P100)
   - **Internet**: On (for checkpoint push/pull)
4. Add Data:
   - Click **+ Add Data** → search `psn2-kaggle` → attach it

### 4. Add Secrets
1. Click the 🔑 **Secrets** panel on the left
2. Add two secrets:
   - `KAGGLE_USERNAME`: your username from kaggle.json
   - `KAGGLE_KEY`: your key from kaggle.json

### 5. Upload Notebook
1. Click **File → Upload Notebook**
2. Select `kaggle_train.ipynb`
3. Or copy-paste the cells manually

### 6. Configure & Run
1. In cell 7 (Session config), set:
   ```python
   STAGE = 'D1'   # D1 | D2 | D3 | D4 | D5 | D6
   ```
2. Click **Run All**

## Subsequent Sessions (Resume Training)

### Automatic Resume (Recommended)
1. Open your notebook from the previous session
2. Click **Edit** (creates a new version)
3. The notebook auto-detects the checkpoint from the previous session
4. Just click **Run All** — it resumes automatically

### Manual Resume (if checkpoint was lost)
1. Download `/kaggle/working/artifacts/latest.pt` from previous session
2. Upload it as a new Kaggle dataset (e.g., `psn2-checkpoint`)
3. Attach `psn2-checkpoint` to your new notebook
4. The notebook will auto-detect and load it

## Stage Progression

| Stage | Focus | Steps | Duration (2×T4) |
|-------|-------|-------|-----------------|
| D1 | Sensorimotor grounding (ARC-AGI-2 + graphs) | 20,000 | ~3-4 hours |
| D2 | Causal grounding | 20,000 | ~3-4 hours |
| D3 | Theory-of-Mind (ToM/ToMi) | 15,000 | ~2-3 hours |
| D4 | Linguistic grounding (Wikitext) | 25,000 | ~4-5 hours |
| D5 | Abstract reasoning (ARC-AGI-2 + GSM8K) | 30,000 | ~5-6 hours |
| D6 | Full integration (BBH) | 40,000 | ~6-8 hours |

**Total**: ~24-30 hours across 6-10 sessions

## Checkpoint Persistence

### How It Works
- Checkpoints save to `/kaggle/working/artifacts/` every 30 min during training
- At session end (or crash), checkpoint is pushed to `psn2-checkpoint` dataset
- Next session pulls it automatically from `/kaggle/input/psn2-checkpoint/`
- No manual download/upload needed (unless you want a backup)

### Crash Recovery
The notebook registers exit handlers that auto-push checkpoints on:
- Normal session end
- Keyboard interrupt (Ctrl+C)
- SIGTERM (Kaggle timeout)
- Python exceptions during training

## Monitoring Progress

### During Training
Watch the console output:
```
step=6000 stage=D1 phase=perceptive 
loss=0.2889 pred=0.8031 shape=0.0016 
nodes=256/256 attractors=2048 goals=50 motifs=0
```

### After Training
Check the evaluation scorecard:
```
[1] Reasoning Integrity
  Grid accuracy:          0.6309  (gate: >= 0.75)
  Relation prediction:    0.4286  (gate: >= 0.85)
  
[5] D1 Gate Status
  [FAIL] object_tracking_accuracy
  [FAIL] causal_prediction_error
  [FAIL] temporal_trace_persistence
  [PASS] vsa_binding_accuracy
```

### Gate Certification
Each stage has 4 gates that must pass before progressing:
- **D1**: object tracking, causal prediction, trace persistence, VSA binding
- **D2**: causal intervention, abstract analogy, VSA causal bonds, compositional split
- **D3**: goal inference, false belief, trust calibration, emotional induction
- **D4**: linguistic fidelity, grounding violations, USL round-trip
- **D5**: transfer learning, meta-learning, compositional generalization
- **D6**: full integration across all subsystems

## Troubleshooting

### "No checkpoint to push"
- Normal on first session — checkpoint is created after first save (30 min or 1000 steps)

### "Kaggle secrets not found"
- Add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Notebook Secrets
- Checkpoint auto-push will be disabled without them (manual backup needed)

### "Could not find train.py"
- The `psn2-kaggle` dataset wasn't attached
- Click **+ Add Data** → search `psn2-kaggle` → attach it

### "CUDA out of memory"
- Reduce `batch_size` from 32 to 16 in the config cell
- Or use single GPU by removing DataParallel (edit train.py)

### "Smoke test failed"
- Check the error message — likely a dataset file is missing
- Verify all data files are present in cell 6 (Verify data)

## Data Files

### Included in `psn2_kaggle_full.zip` (3.8 MB)
- `data/d5_arc_agi2/` — ARC-AGI-2 grids (1000 train, 120 eval)
- `data/d3_tom/` — Theory-of-Mind stories (13k samples)
- `data/d3_tomi/` — ToMi entailment (18k samples)
- `data/d5_gsm8k/` — Grade-school math (7.5k train, 1.3k test)
- `data/d6_bbh/` — BIG-Bench Hard (6.5k test)

### Optional (upload separately)
- `data/d4_wikitext/` — Wikitext-103 (550 MB, 1.8M lines)
  - Only needed for D4 stage
  - Falls back to synthetic data if missing

## Cost Estimate

- **Kaggle Free Tier**: 30 GPU hours/week
- **PSN-2 Full Training**: ~24-30 hours total
- **Strategy**: Run 1-2 stages per week to stay within free tier
- **Alternative**: Kaggle Notebooks+ ($20/month) for unlimited GPU

## Next Steps After Training

1. **Download final checkpoint**: `/kaggle/working/artifacts/final_*.pt`
2. **Download eval results**: `/kaggle/working/eval_results.json`
3. **Run local evaluation**: `python evaluate.py --checkpoint final_*.pt`
4. **Test on ARC-AGI-2 eval set**: Use the 120 held-out tasks
5. **Submit to ARC Prize**: If D1-D6 gates all pass

## Support

- **Issues**: [github.com/FPC-effortless/PSN2/issues](https://github.com/FPC-effortless/PSN2/issues)
- **PRD**: See `PRD.md` for full architecture details
- **Changelog**: See `CHANGELOG.md` for recent fixes
