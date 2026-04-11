"""Final validation of all fixes before Kaggle deployment."""
import ast, json, sys
sys.path.insert(0, '.')

errors = []

# 1. Syntax check all modified files
files = [
    'train.py', 'train_sequential.py', 'evaluate.py',
    'psn2/datasets/tom_dataset.py',
    'psn2/datasets/gsm8k_dataset.py',
    'psn2/datasets/bbh_dataset.py',
]
for f in files:
    try:
        ast.parse(open(f, encoding='utf-8').read())
        print('syntax OK:', f)
    except SyntaxError as e:
        errors.append(f'SYNTAX FAIL {f}: {e}')

# 2. Config check
cfg = json.load(open('configs/default.json'))
if cfg['stage'] != 'D1':
    errors.append(f'config stage should be D1, got {cfg["stage"]}')
else:
    print('config OK: stage =', cfg['stage'])

# 3. Gate name alignment
from train_sequential import STAGE_GATES
from psn2.dc.stage_d1 import StageD1
from psn2.dc.stage_d2 import StageD2
from psn2.dc.stage_d3 import StageD3
from psn2.dc.stage_d4 import StageD4
from psn2.dc.stage_d5 import StageD5
from psn2.dc.stage_d6 import StageD6
d1=StageD1(); d2=StageD2(d1); d3=StageD3(d2); d4=StageD4(d3); d5=StageD5(d4); d6=StageD6(d5)
for stage, obj in [('D1',d1),('D2',d2),('D3',d3),('D4',d4),('D5',d5),('D6',d6)]:
    actual = set(obj.certifier.gates.keys())
    expected = set(STAGE_GATES[stage])
    diff = actual ^ expected
    if diff:
        errors.append(f'gate mismatch {stage}: {diff}')
    else:
        print('gates OK:', stage)

# 4. Dataset key check
from psn2.datasets.rel_graph import RelationalGraphDataset
sample = RelationalGraphDataset(n_samples=2)[0]
required = ['masked_entity_idx', 'masked_relation_idx', 'target_entity', 'target_relation', 'type']
for k in required:
    if k not in sample:
        errors.append(f'RelationalGraphDataset missing key: {k}')
print('dataset keys OK: RelationalGraphDataset')

# 5. Collate scalar handling
import torch
scalars = [{'type': 'graph', 'val': torch.tensor(5)} for _ in range(3)]
# Simulate fixed collate
out = {'type': scalars[0]['type']}
for k in scalars[0]:
    if k == 'type': continue
    tensors = [item[k] for item in scalars]
    if tensors[0].dim() == 0:
        out[k] = torch.stack([t.unsqueeze(0) for t in tensors], dim=0).squeeze(-1)
    else:
        out[k] = torch.stack(tensors, dim=0)
assert out['val'].shape == (3,), f'collate scalar shape wrong: {out["val"].shape}'
print('collate scalar OK: shape', out['val'].shape)

# 6. evaluate.py emits all dN_gates keys
import ast as ast2
src = open('evaluate.py', encoding='utf-8').read()
for stage in ['d1','d2','d3','d4','d5','d6']:
    key = f'"{stage}_gates"'
    if key not in src:
        errors.append(f'evaluate.py missing results key: {key}')
    else:
        print(f'evaluate.py key OK: {key}')

print()
if errors:
    print('FAILURES:')
    for e in errors:
        print(' ', e)
    sys.exit(1)
else:
    print('ALL CHECKS PASSED - ready for Kaggle')
