"""Quick test of all dataset loaders."""
from psn2.datasets import (
    ARCGridDataset, RelationalGraphDataset,
    ARCAGI2Dataset, ToMDataset, ToMiDataset,
    WikitextDataset, GSM8KDataset, BBHDataset,
)

ds = ARCAGI2Dataset('data/d5_arc_agi2/train.jsonl', max_grid_size=8, max_samples=10)
s = ds[0]
assert s['type'] == 'arc'
assert s['input_grid'].shape == (8, 8)
print(f'ARCAGI2Dataset: {len(ds)} samples, grid {s["input_grid"].shape}')

ds = ToMDataset('data/d3_tom/train.jsonl', max_samples=10)
s = ds[0]
assert s['type'] == 'graph' and s['entities'].shape == (6,)
print(f'ToMDataset: {len(ds)} samples')

ds = ToMiDataset('data/d3_tomi/train.jsonl', max_samples=10)
s = ds[0]
assert s['target_entity'].item() in (0, 1)
print(f'ToMiDataset: {len(ds)} samples')

ds = WikitextDataset('data/d4_wikitext/train.jsonl', grid_size=8, max_samples=20)
s = ds[0]
assert s['input_grid'].shape == (8, 8)
print(f'WikitextDataset: {len(ds)} samples')

ds = GSM8KDataset('data/d5_gsm8k/train.jsonl', max_samples=10)
s = ds[0]
assert s['type'] == 'graph'
print(f'GSM8KDataset: {len(ds)} samples')

ds = BBHDataset('data/d6_bbh/test.jsonl', max_samples=10)
s = ds[0]
assert s['type'] == 'graph'
print(f'BBHDataset: {len(ds)} samples')

print('\nAll dataset loaders OK.')
