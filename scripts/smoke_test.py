"""End-to-end smoke test — runs 3 steps of training for each stage using real data."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from psn2.core import PSN2System
from psn2.datasets import (
    ARCGridDataset, RelationalGraphDataset,
    ARCAGI2Dataset, ToMDataset, ToMiDataset,
    WikitextDataset, GSM8KDataset, BBHDataset,
)

def collate(batch):
    out = {"type": batch[0]["type"]}
    for k in batch[0].keys():
        if k == "type":
            continue
        out[k] = torch.stack([item[k] for item in batch], dim=0)
    return out

def run_steps(model, loader, n=3, phase="perceptive"):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    it = iter(loader)
    for i in range(n):
        batch = next(it)
        out = model.forward_batch(batch, phase=phase)
        out["loss"].backward()
        opt.step(); opt.zero_grad()
    return out["loss"].item()

print("Smoke test: imports OK")
print(f"  ARCGridDataset:    {len(ARCGridDataset(10))} samples")
print(f"  RelGraphDataset:   {len(RelationalGraphDataset(10))} samples")

# Real data loaders
data_dir = "data"
tests = []

arc2_path = f"{data_dir}/d5_arc_agi2/train.jsonl"
if os.path.exists(arc2_path):
    ds = ARCAGI2Dataset(arc2_path, max_grid_size=8, max_samples=20)
    tests.append(("ARCAGI2Dataset (D1/D5)", ds))

tom_path = f"{data_dir}/d3_tom/train.jsonl"
if os.path.exists(tom_path):
    ds = ToMDataset(tom_path, max_samples=20)
    tests.append(("ToMDataset (D3)", ds))

tomi_path = f"{data_dir}/d3_tomi/train.jsonl"
if os.path.exists(tomi_path):
    ds = ToMiDataset(tomi_path, max_samples=20)
    tests.append(("ToMiDataset (D3)", ds))

    wiki_path = f"{data_dir}/d4_wikitext/train.jsonl"
if os.path.exists(wiki_path):
    ds = WikitextDataset(wiki_path, vocab_size=10, grid_size=8, max_samples=20)
    tests.append(("WikitextDataset (D4)", ds))

gsm_path = f"{data_dir}/d5_gsm8k/train.jsonl"
if os.path.exists(gsm_path):
    ds = GSM8KDataset(gsm_path, max_samples=20)
    tests.append(("GSM8KDataset (D5)", ds))

bbh_path = f"{data_dir}/d6_bbh/test.jsonl"
if os.path.exists(bbh_path):
    ds = BBHDataset(bbh_path, max_samples=20)
    tests.append(("BBHDataset (D6)", ds))

model = PSN2System(dim=64, max_nodes=16, grid_vocab=10, rel_vocab=64, stage="D1")
print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

for name, ds in tests:
    loader = DataLoader(ds, batch_size=4, collate_fn=collate, drop_last=True)
    loss = run_steps(model, loader, n=3)
    print(f"  {name}: loss={loss:.4f} OK")

print("\nAll smoke tests passed.")
