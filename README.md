# PSN-2 Kaggle-Constrained Full Scaffold

This repository is a runnable scaffold for a Kaggle-sized PSN-2 implementation.

It includes:
- Bipolar VSA substrate
- Sparse spiking node layer
- Mixed D1/D2 datasets
  - ARC-style grid completion
  - Synthetic relational graphs
- Mixed training loop
- Growth ledger
- Attractor, motif, and curiosity hooks
- Checkpoint save/load
- Evaluation script

## Run

```bash
pip install -r requirements.txt
python train.py --config configs/default.json
python evaluate.py --config configs/default.json --checkpoint artifacts/latest.pt
```

## Notes

This is a scaffold intended to be extended into a research system.
