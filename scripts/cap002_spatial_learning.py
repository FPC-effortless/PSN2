"""CAP002: bounded synthetic spatial learning, not ARC-AGI or superiority.

18 predeclared fits. Actual PSN2 source remains unchanged. Full native forward
is compared with its own encoder/decoder using zero global context. Primary
CE only; optional heads are not expected to receive gradients. All targets
are scored externally at evaluation. No search over held-out configurations.
"""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from psn2.core import PSN2System

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else 'ci-artifacts/cap002')
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [101, 202, 303]
TASKS = ['recolor', 'shift_right', 'reflection']
ARMS = ['native', 'decoder_only']
STEPS, BATCH, VOCAB = 200, 16, 4

def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

def create(s):
    seed_all(s)
    return PSN2System(dim=32, max_nodes=16, grid_vocab=VOCAB,
                      rel_vocab=5, stage='D1')

def grids(n, size, seed):
    a = np.random.default_rng(seed).integers(0, VOCAB, (n, size, size), dtype=np.int64)
    return torch.from_numpy(a)

DATA = {'fit': grids(1024, 4, 41001), 'eval': grids(64, 4, 51001),
        'size_shift': grids(64, 6, 61001)}

def row_hashes(x):
    return {hashlib.sha256(str(tuple(v.shape)).encode() + v.numpy().tobytes()).hexdigest()
            for v in x}

def targets(x, task):
    if task == 'recolor':
        return (x + 1) % VOCAB
    if task == 'reflection':
        return x.flip(-1)
    y = torch.zeros_like(x)
    y[:, :, 1:] = x[:, :, :-1]
    return y

def logits(m, x, y, arm):
    if arm == 'decoder_only':
        return m.grid_decoder(torch.zeros((len(x), m.dim)), m.grid_encoder(x), input_grid=x)
    captured = []
    hook = m.grid_decoder.register_forward_hook(lambda module, args, output: captured.append(output))
    try:
        m.forward_batch({'type': 'arc', 'input_grid': x, 'target_grid': y})
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError(f'Expected one actual decoder call, observed {len(captured)}')
    return captured[0]

def snapshot(m):
    b = io.BytesIO()
    torch.save(m.state_dict_full(), b)
    return b.getvalue()

def restore(m, data):
    m.load_state_dict_full(torch.load(io.BytesIO(data), map_location='cpu', weights_only=False))

def metrics(pred, y, ce):
    equal = pred.eq(y)
    return {'n_grids': len(y), 'n_cells': y.numel(),
            'cell_accuracy': equal.float().mean().item(),
            'exact_grid_accuracy': equal.flatten(1).all(1).float().mean().item(),
            'cross_entropy': float(ce)}

def evaluate(state, arm, x, task, seed, prefix):
    # Every query starts from the same trained model state; no cross-query carry.
    # Batch size one eliminates companion-example information at evaluation.
    m = create(seed).eval()
    predictions, losses = [], []
    with torch.no_grad():
        for i, row in enumerate(x):
            restore(m, state)
            seed_all(seed + 900000 + i)
            inp = row.unsqueeze(0)
            p = logits(m, inp, torch.zeros_like(inp), arm)
            y = targets(inp, task)
            losses.append(F.cross_entropy(p.reshape(-1, VOCAB), y.reshape(-1)).item())
            predictions.append(p.argmax(-1))
    pred = torch.cat(predictions)
    y = targets(x, task)
    np.savez_compressed(OUT / f'{prefix}.npz', inputs=x.numpy(), targets=y.numpy(), predictions=pred.numpy())
    return metrics(pred, y, np.mean(losses))

def preflight():
    hashes = {k: row_hashes(v) for k, v in DATA.items()}
    disjoint = all(not (hashes[a] & hashes[b]) for a, b in [('fit', 'eval'), ('fit', 'size_shift'), ('eval', 'size_shift')])
    unique = all(len(hashes[k]) == len(v) for k, v in DATA.items())
    x = DATA['fit'][:2]
    a, b = create(101).eval(), create(101).eval()
    with torch.no_grad():
        p = logits(a, x, torch.zeros_like(x), 'native')
        q = logits(b, x, torch.full_like(x, 3), 'native')
    poison_delta = (p - q).abs().max().item()
    # Companion dependence is a diagnostic, not mistaken for label leakage.
    a, b = create(101).eval(), create(101).eval()
    with torch.no_grad():
        p = logits(a, x, torch.zeros_like(x), 'native')[0]
        altered = x.clone(); altered[1] = (altered[1] + 1) % VOCAB
        q = logits(b, altered, torch.zeros_like(altered), 'native')[0]
    batch_delta = (p - q).abs().max().item()
    checks = {'split_disjoint': disjoint, 'within_split_unique': unique,
              'target_poison_max_logit_delta': poison_delta,
              'companion_change_max_logit_delta': batch_delta,
              'evaluation_policy': 'batch_one_restore_identical_trained_full_state_per_query_dummy_targets',
              'data_sha256': {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in DATA.items()}}
    if not disjoint or not unique or poison_delta > 1e-6:
        raise RuntimeError('Integrity preflight failed: ' + json.dumps(checks))
    return checks

def fit(task, arm, seed):
    name = f'{task}_{arm}_{seed}'
    t0 = time.monotonic()
    m = create(seed)
    params = list(m.parameters())
    parameter_count = sum(p.numel() for p in params)
    opt = torch.optim.Adam(params, lr=0.003)
    initial = snapshot(m)
    before = evaluate(initial, arm, DATA['eval'], task, seed, name + '_before')
    order = np.random.default_rng(seed + 97).integers(0, len(DATA['fit']), (STEPS, BATCH))
    np.save(OUT / f'{name}_batch_order.npy', order)
    trace, connected = [], None
    # Evaluation operates on separate instances; restore RNG before training.
    seed_all(seed + 1000)
    m.train()
    done = 0
    for step, ids in enumerate(order):
        x = DATA['fit'][torch.from_numpy(ids)]
        y = targets(x, task)
        opt.zero_grad(set_to_none=True)
        p = logits(m, x, y, arm)
        loss = F.cross_entropy(p.reshape(-1, VOCAB), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        if connected is None:
            connected = {'connected': [n for n, v in m.named_parameters() if v.grad is not None],
                         'disconnected': [n for n, v in m.named_parameters() if v.grad is None],
                         'connected_parameters': sum(v.numel() for v in m.parameters() if v.grad is not None)}
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in params):
            raise RuntimeError(f'Nonfinite gradient at step {step}')
        opt.step()
        done += 1
        trace.append({'step': done, 'primary_ce': loss.item()})
        # Progress is durable even if an attempt fails later.
        if done % 50 == 0:
            print(json.dumps({'progress': name, 'optimizer_steps': done, 'ce': loss.item()}), flush=True)
            (OUT / f'{name}_trace.json').write_text(json.dumps(trace))
    state = snapshot(m)
    torch.save({'study': 'CAP002', 'task': task, 'arm': arm, 'seed': seed,
                'full_state_bytes': state, 'optimizer': opt.state_dict(),
                'steps': done, 'batch_order': order, 'python_rng': random.getstate(),
                'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state()}, OUT / f'{name}.pt')
    result = {'task': task, 'arm': arm, 'seed': seed, 'status': 'MEASURED',
              'optimizer_steps': done, 'total_parameters': parameter_count,
              'primary_gradients_first_step': connected, 'before': before,
              'eval': evaluate(state, arm, DATA['eval'], task, seed, name + '_eval'),
              'size_shift': evaluate(state, arm, DATA['size_shift'], task, seed, name + '_size_shift')}
    if arm == 'native':
        result['zero_context_intervention'] = evaluate(state, 'decoder_only', DATA['eval'], task, seed, name + '_zero_context')
    # Verify next-forward replay, explicitly not full training-continuation replay.
    a, b = create(seed).eval(), create(seed).eval()
    restore(a, state); restore(b, state)
    x = DATA['eval'][:1]
    with torch.no_grad():
        seed_all(seed + 777); p = logits(a, x, torch.zeros_like(x), arm)
        seed_all(seed + 777); q = logits(b, x, torch.zeros_like(x), arm)
    result['model_only_replay_max_delta'] = (p - q).abs().max().item()
    result['wall_seconds'] = time.monotonic() - t0
    result['train_trace'] = trace
    (OUT / f'{name}_result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ['train_trace', 'primary_gradients_first_step']}), flush=True)
    return result

report = {'study': 'CAP002', 'qualification': 'SYNTHETIC_EXPLORATORY_LEARNING_NOT_GENERAL_BENCHMARK',
          'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'source_core_sha256': hashlib.sha256((ROOT / 'psn2/core.py').read_bytes()).hexdigest(),
          'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version()},
          'seeds': SEEDS, 'planned_fits': 18, 'steps_per_fit': STEPS, 'batch_size': BATCH,
          'new_gpu_minutes': 0, 'architecture_superiority_eligible': False,
          'replay_scope': 'model-only identical-next-forward; optimizer/RNG saved but fresh-process continuation NOT_RUN',
          'rows': []}
try:
    report['integrity'] = preflight()
except Exception:
    report['integrity_error'] = traceback.format_exc()
    (OUT / 'RESULTS.json').write_text(json.dumps(report, indent=2))
    raise
report['copy_baselines'] = {task: {split: metrics(x, targets(x, task), float('nan'))
                                  for split, x in DATA.items() if split != 'fit'} for task in TASKS}
# Replace nonfinite placeholder by JSON null; copy rule has no probability model.
for task in report['copy_baselines'].values():
    for split in task.values():
        split['cross_entropy'] = None
for task in TASKS:
    for seed in SEEDS:
        for arm in ARMS:
            try:
                row = fit(task, arm, seed)
            except Exception as exc:
                row = {'task': task, 'arm': arm, 'seed': seed, 'status': 'ERROR',
                       'error': str(exc), 'traceback': traceback.format_exc()}
                print(json.dumps(row), flush=True)
            report['rows'].append(row)
            (OUT / 'RESULTS.json').write_text(json.dumps(report, indent=2, allow_nan=False))
report['measured_fits'] = sum(r['status'] == 'MEASURED' for r in report['rows'])
report['error_fits'] = sum(r['status'] == 'ERROR' for r in report['rows'])
(OUT / 'RESULTS.json').write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps({'summary': {'measured': report['measured_fits'], 'errors': report['error_fits']}}), flush=True)
if report['error_fits']:
    raise SystemExit(1)
