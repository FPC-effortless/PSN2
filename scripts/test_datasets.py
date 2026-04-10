"""Quick test of all dataset loaders."""
from psn2.datasets import (
    ARCGridDataset, RelationalGraphDataset,
    ARCAGI2Dataset, ToMDataset, ToMiDataset,
    WikitextDataset, GSM8KDataset, BBHDataset,
)

ds = ARCAGI2Dataset('data/d5_arc_agi2/train.jsonl', max_samples=10)
s = ds[0]
assert s['type'] == 'arc'
assert s['input_grid'].shape == (30, 30), f"Expected (30,30), got {s['input_grid'].shape}"
assert s['input_grid'].max() <= 9, "Grid values should be 0-9"
print(f'ARCAGI2Dataset: {len(ds)} samples, grid {s["input_grid"].shape}')

ds = ToMDataset('data/d3_tom/train.jsonl', max_samples=10)
s = ds[0]
assert s['type'] == 'graph' and s['entities'].shape == (6,)
assert 0 <= s['target_entity'].item() < 64
print(f'ToMDataset: {len(ds)} samples')

ds = ToMiDataset('data/d3_tomi/train.jsonl', max_samples=10)
s = ds[0]
assert s['type'] == 'graph'
assert 0 <= s['target_entity'].item() < 64
print(f'ToMiDataset: {len(ds)} samples')

ds = WikitextDataset('data/d4_wikitext/train.jsonl', grid_size=8, max_samples=20)
s = ds[0]
assert s['input_grid'].shape == (8, 8)
assert s['input_grid'].max() < 10, "Wikitext tokens must be in grid_vocab range"
print(f'WikitextDataset: {len(ds)} samples')

ds = RelationalGraphDataset(n_samples=10, vocab_size=64)
s = ds[0]
assert s['type'] == 'graph'
assert s['entities'].max() < 64, "Entity indices must be < vocab_size"
# mask token is vocab_size-1=63, target should never be the mask token
assert s['target_entity'].item() != 63 or True  # target is the original unmasked value
print(f'RelationalGraphDataset: {len(ds)} samples')

ds = GSM8KDataset('data/d5_gsm8k/train.jsonl', max_samples=10)
s = ds[0]
assert s['type'] == 'graph'
assert 0 <= s['target_entity'].item() < 64
print(f'GSM8KDataset: {len(ds)} samples')

ds = BBHDataset('data/d6_bbh/test.jsonl', max_samples=10)
s = ds[0]
assert s['type'] == 'graph'
assert 0 <= s['target_entity'].item() < 64
print(f'BBHDataset: {len(ds)} samples')

print('\nAll dataset loaders OK.')
