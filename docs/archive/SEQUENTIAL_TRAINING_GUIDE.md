# Sequential Training Guide - PSN-2

## Overview

Train all developmental stages (D1→D2→D3→D4→D5→D6) automatically with gate certification between stages.

## Two Ways to Use Sequential Training

### Option 1: Local Training (Recommended for Development)

```bash
python train_sequential.py --config configs/default.json
```

This will:
1. Train D1 for 20,000 steps
2. Evaluate and check D1 gates
3. If gates pass → train D2 for 20,000 steps
4. Continue through D6
5. Stop if any stage fails gate certification

### Option 2: Kaggle Training (Recommended for Production)

Use `kaggle_train_sequential.ipynb` - see instructions below.

## Local Training Options

### Basic Usage
```bash
# Train all stages D1→D6
python train_sequential.py

# Train specific range
python train_sequential.py --start-stage D2 --end-stage D4

# Resume from checkpoint
python train_sequential.py --resume checkpoints_sequential/latest.pt

# Continue even if gates fail
python train_sequential.py --skip-gate-check

# Force continue on errors
python train_sequential.py --force-continue
```

### Full Options
```bash
python train_sequential.py \
  --config configs/default.json \
  --checkpoint-dir checkpoints_sequential \
  --start-stage D1 \
  --end-stage D6 \
  --resume checkpoints_sequential/latest.pt \
  --skip-gate-check \
  --force-continue
```

## Kaggle Sequential Training

### Setup (One Time)

1. **Upload Dataset**
   - Create `psn2_kaggle_full_v2.zip` (includes `train_sequential.py`)
   - Upload to Kaggle as `psn2-kaggle` dataset

2. **Create Notebook**
   - Upload `kaggle_train_sequential.ipynb`
   - Set GPU to **2×T4** or **P100**
   - Attach `psn2-kaggle` dataset

3. **Configure Secrets**
   - Add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Notebook Secrets

### Running Sequential Training on Kaggle

**Cell 6 Configuration:**
```python
START_STAGE = 'D1'   # Start from D1
END_STAGE   = 'D6'   # Train through D6
SKIP_GATE_CHECK = False  # Stop if gates fail
```

**Then click "Run All"**

### Training Strategies

#### Strategy 1: One Stage Per Session (Free Tier)
```python
# Session 1
START_STAGE = 'D1'
END_STAGE   = 'D1'

# Session 2 (after D1 passes)
START_STAGE = 'D2'
END_STAGE   = 'D2'

# ... continue through D6
```

**Pros:** Fits in Kaggle free tier (30 GPU hours/week)
**Cons:** Manual intervention between stages

#### Strategy 2: Continuous Training (Notebooks+)
```python
START_STAGE = 'D1'
END_STAGE   = 'D6'
SKIP_GATE_CHECK = False  # Stop if any stage fails
```

**Pros:** Fully automated, no manual intervention
**Cons:** Requires Kaggle Notebooks+ ($20/month) for extended runtime

#### Strategy 3: Checkpoint and Resume
```python
# Session 1: Train D1-D3
START_STAGE = 'D1'
END_STAGE   = 'D3'

# Session 2: Resume from D4
START_STAGE = 'D4'
END_STAGE   = 'D6'
```

**Pros:** Balances automation and free tier limits
**Cons:** Need to manually set start stage

## Stage Details

| Stage | Focus | Steps | Time (2×T4) | Gates |
|-------|-------|-------|-------------|-------|
| D1 | Sensorimotor grounding | 20,000 | ~3-4h | 4 gates |
| D2 | Causal grounding | 20,000 | ~3-4h | 4 gates |
| D3 | Theory-of-Mind | 15,000 | ~2-3h | 4 gates |
| D4 | Linguistic grounding | 25,000 | ~4-5h | 3 gates |
| D5 | Abstract reasoning | 30,000 | ~5-6h | 3 gates |
| D6 | Full integration | 40,000 | ~6-8h | 1 gate |

**Total:** 150,000 steps, ~24-30 hours

## Gate Certification

Each stage has gates that must pass before continuing:

### D1 Gates
- ✓ Object tracking accuracy (≥75%)
- ✓ Causal prediction error (<15%)
- ✓ Temporal trace persistence (>5 pulses)
- ✓ VSA binding accuracy (≥85%)

### D2 Gates
- ✓ Causal intervention accuracy
- ✓ Abstract analogy score
- ✓ VSA causal bond recall
- ✓ Compositional split performance

### D3 Gates
- ✓ Goal inference accuracy
- ✓ False belief task performance
- ✓ Trust calibration score
- ✓ Emotional induction accuracy

### D4 Gates
- ✓ Linguistic fidelity (USL round-trip)
- ✓ Grounding violations (low rate)
- ✓ USL round-trip fidelity

### D5 Gates
- ✓ Transfer learning efficiency
- ✓ Meta-learning performance
- ✓ Compositional generalization

### D6 Gates
- ✓ Full integration across all subsystems

## Output Files

Sequential training creates these files in `checkpoint_dir`:

```
checkpoints_sequential/
├── latest.pt                    # Current checkpoint
├── config_D1.json              # D1 config
├── config_D2.json              # D2 config
├── ...
├── eval_D1.json                # D1 evaluation results
├── eval_D2.json                # D2 evaluation results
├── ...
├── sequential_results.json     # Summary of all stages
└── final_D6_step_150000.pt    # Final checkpoint (if D6 completes)
```

## Monitoring Progress

### During Training
Watch for stage transitions:
```
======================================================================
STAGE D1
======================================================================

Stage: D1
Steps: 20000
...

[D1] Starting training...
step=1000 stage=D1 phase=perceptive loss=0.2889 ...
...

[D1] Evaluating...
[D1] Gate Certification:
  ✓ All gates PASSED for D1

======================================================================
STAGE D2
======================================================================
```

### After Training
Check `sequential_results.json`:
```json
{
  "D1": {
    "passed": true,
    "eval_results": {
      "grid_accuracy": 0.8070,
      "relation_prediction": 0.7850,
      ...
    }
  },
  "D2": {
    "passed": true,
    ...
  }
}
```

## Troubleshooting

### "Stage failed gates"
```bash
# Option 1: Continue anyway
python train_sequential.py --skip-gate-check

# Option 2: Retrain just that stage
python train_sequential.py --start-stage D2 --end-stage D2

# Option 3: Investigate
cat checkpoints_sequential/eval_D2.json
```

### "Training interrupted"
```bash
# Resume from where it left off
python train_sequential.py --resume checkpoints_sequential/latest.pt
```

The checkpoint includes the current stage, so it will resume from the right place.

### "Out of memory"
Edit the config to reduce batch size:
```json
{
  "batch_size": 16  // was 32
}
```

### "Kaggle session timeout"
The checkpoint is automatically pushed on timeout. Next session:
1. Attach the `psn2-checkpoint` dataset
2. Set `START_STAGE` to the next stage
3. Run all cells

## Comparison: Single vs Sequential

### Single Stage Training (`train.py`)
```bash
# Train D1
python train.py --config configs/default.json

# Manually evaluate
python evaluate.py --checkpoint checkpoints/latest.pt

# Manually check gates
# ... inspect results ...

# Manually train D2
# ... edit config to stage=D2 ...
python train.py --config configs/default.json --resume checkpoints/latest.pt
```

**Pros:** Fine-grained control
**Cons:** Manual intervention, easy to forget steps

### Sequential Training (`train_sequential.py`)
```bash
# Train all stages
python train_sequential.py
```

**Pros:** Fully automated, gate checks built-in, no manual steps
**Cons:** Less control over individual stages

## Best Practices

1. **Start with D1 only** to verify setup:
   ```bash
   python train_sequential.py --start-stage D1 --end-stage D1
   ```

2. **Use `--skip-gate-check` for exploration** but not for final training

3. **Monitor the first few hundred steps** of each stage to catch issues early

4. **Save intermediate checkpoints** by copying `latest.pt` after each stage

5. **Check gate failures immediately** - don't wait until the end

## Next Steps

After successful sequential training:

1. **Download final checkpoint**
   ```bash
   # Local
   cp checkpoints_sequential/final_D6_step_150000.pt ./final_model.pt
   
   # Kaggle
   # Download from /kaggle/working/artifacts/final_D6_step_150000.pt
   ```

2. **Run comprehensive evaluation**
   ```bash
   python evaluate.py --checkpoint final_model.pt --output final_eval.json
   ```

3. **Test on held-out data**
   - ARC-AGI-2 evaluation set (120 tasks)
   - Custom test cases

4. **Deploy or submit**
   - ARC Prize submission
   - Production deployment
   - Further fine-tuning

## Support

- See `RELATION_PREDICTION_FIX.md` for recent fixes
- See `KAGGLE_GUIDE.md` for Kaggle-specific help
- See `PRD.md` for architecture details
