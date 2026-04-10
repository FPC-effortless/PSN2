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

### Package the repo (run once locally)
```bash
python scripts/package_kaggle.py
# → psn2_kaggle_full.zip (~4MB, wikitext excluded)
```

### One-time Kaggle setup
1. [kaggle.com/datasets](https://kaggle.com/datasets) → **New Dataset** → upload `psn2_kaggle_full.zip` → name it `psn2-kaggle`
2. **Account → Settings → API → Create New Token** → download `kaggle.json`
3. In your notebook: **Add-ons → Secrets** → add `KAGGLE_USERNAME` and `KAGGLE_KEY`
4. Create a **Notebook** → attach `psn2-kaggle` → **Settings → Accelerator → GPU T4 x2**
5. Upload `kaggle_train.ipynb`, set `STAGE = 'D1'`, run all cells

### Subsequent sessions — fully automatic
- The notebook pushes `latest.pt` to a `psn2-checkpoint` Kaggle dataset at session end
- On crash or error it pushes immediately via an exit handler
- Next session: attach both `psn2-kaggle` and `psn2-checkpoint` → run all cells → auto-resumes
- No manual download/upload needed

### Session plan (PRD Section 17.3)

| Session | Stage | Steps | Data |
|---------|-------|-------|------|
| 1–2 | D1 | 20,000 | ARC-AGI-2 + synthetic graphs |
| 3–4 | D2 | 20,000 | ARC-AGI-2 causal + synthetic |
| 5 | D3 | 15,000 | ToM + ToMi |
| 6–7 | D4 | 25,000 | Wikitext + ARC-AGI-2 |
| 8–9 | D5 | 30,000 | ARC-AGI-2 + GSM8K |
| 10+ | D6 | 40,000 | All data + BBH |

### Wikitext (D4)
Wikitext is 550MB — excluded from the main zip. Upload `data/d4_wikitext/` as a separate Kaggle dataset and attach it. Without it, D4 falls back to synthetic ARC data.

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
