# PSN-2 — Predictive Substrate Network Generation 2

Unified cognitive architecture implementing the PSN-2 PRD v1.0.
Designed to train across constrained Kaggle sessions (T4/P100) with checkpoint-driven growth.

## Architecture

- **VSA substrate** — bind/bundle/cleanup hypervector algebra (D=512 Lite)
- **Spiking Computation Engine (SCE)** — event-driven sparse inference
- **6-phase pulse cycle** — A (evidence) → B (bonds) → C (emotion) → D (shape) → E (verify) → F (commit)
- **Substrate Growth Protocol (SGP)** — spawn/prune/merge/expand with I-24 enforcement
- **Experience Replay Substrate (ERS)** — 4-tier VSA memory
- **Curiosity Engine (CE)** — goal-directed exploration with flood protection
- **Unified Shape Language (USL)** — language as native VSA shapes
- **Emotional Shape System (ESS)** — emotion as first-class shapes
- **Temporal Abstraction Engine (TAE)** — motif compression
- **Social Learning System (SLS)** — trace reconstruction
- **Developmental Curriculum (DC)** — D1→D6 gated stages

## Training Data (collected in `data/`)

| Stage | Dataset | Samples | Purpose |
|-------|---------|---------|---------|
| D1/D2/D5 | ARC-AGI-2 (`d5_arc_agi2/`) | 1,000 tasks | Spatial/causal grounding |
| D3 | ToM (`d3_tom/`) | 13,309 | Theory-of-mind |
| D3 | ToMi (`d3_tomi/`) | 5,994 | False-belief NLI |
| D4 | Wikitext (`d4_wikitext/`) | 1.8M lines | Linguistic grounding |
| D5 | GSM8K (`d5_gsm8k/`) | 7,473 | Math reasoning |
| D6 | BIG-Bench Hard (`d6_bbh/`) | 6,511 | Abstract reasoning |

## Local Training

```bash
pip install -r requirements.txt

# D1 training (ARC-AGI-2 + synthetic graphs)
python train.py --config configs/default.json

# Resume from checkpoint
python train.py --config configs/default.json --resume artifacts/latest.pt

# Evaluate
python evaluate.py --config configs/default.json --checkpoint artifacts/latest.pt
```

## Kaggle Training

### One-time setup
```bash
# Package everything (run from repo root)
bash scripts/package_kaggle.sh
# → creates psn2_kaggle_full.zip (~50MB without wikitext)
```

### Per session
1. Go to [kaggle.com/datasets](https://kaggle.com/datasets) → **New Dataset** → upload `psn2_kaggle_full.zip`
2. Create a new **Notebook** → attach the dataset
3. **Settings → Accelerator → GPU T4 x2** (or P100)
4. Upload `kaggle_train.ipynb` or paste its cells
5. Set `STAGE = 'D1'` in the config cell (change per session)
6. Run all cells

### Session plan (PRD Section 17.3)

| Session | Stage | Steps | Data |
|---------|-------|-------|------|
| 1–2 | D1 | 20,000 | ARC-AGI-2 + synthetic graphs |
| 3–4 | D2 | 20,000 | ARC-AGI-2 causal + synthetic |
| 5 | D3 | 15,000 | ToM + ToMi |
| 6–7 | D4 | 25,000 | Wikitext + ARC-AGI-2 |
| 8–9 | D5 | 30,000 | ARC-AGI-2 + GSM8K |
| 10+ | D6 | 40,000 | All data + BBH |

### Resuming across sessions
- Download `artifacts/latest.pt` from the previous session's output
- Upload it as a **separate dataset attachment** in the next session
- The notebook auto-detects and resumes from it

### Wikitext (D4)
Wikitext is 550MB — too large for the main zip. Options:
- Upload `data/d4_wikitext/` as a separate Kaggle dataset
- Set `cfg['data_dir']` to its mount path in the notebook config cell
- Without it, D4 falls back to synthetic ARC data

## Tests

```bash
python -m pytest tests/ -q
```

37 tests covering: ERS tiers, phase cycle, growth/prune/merge, VSA cosine, bonds, attractor cache, forward pass, checkpoint round-trip.

## Config

All PRD-frozen constants are in `configs/default.json`. Key Kaggle settings:

```json
{
  "vsa_dim": 512,
  "max_nodes": 256,
  "B_max": 32,
  "stage": "D1",
  "data_dir": "data",
  "batch_size": 16,
  "steps": 2000
}
```
