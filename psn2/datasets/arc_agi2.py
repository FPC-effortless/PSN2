"""ARC-AGI-2 dataset loader — D1/D5 stage data.

Each task has 2-10 demonstration pairs (input grid → output grid) and one
test input. We treat each demonstration pair as a training sample.
Grids are variable-size; we pad to max_size and encode cell values as integers.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _pad_grid(grid: List[List[int]], max_h: int, max_w: int) -> torch.Tensor:
    """Pad a variable-size grid to (max_h, max_w) with zeros."""
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    t = torch.zeros(max_h, max_w, dtype=torch.long)
    for r in range(min(h, max_h)):
        for c in range(min(w, max_w)):
            t[r, c] = grid[r][c]
    return t


def _grid_size(grid: List[List[int]]):
    return len(grid), len(grid[0]) if grid else 0


class ARCAGI2Dataset(Dataset):
    """
    Loads ARC-AGI-2 tasks from a JSONL file.
    Each sample is one (input_grid, target_grid) demonstration pair.
    Grids are padded to (max_grid_size, max_grid_size).
    """

    def __init__(self, path: str, max_grid_size: int = 30, max_samples: Optional[int] = None):
        self.max_grid_size = max_grid_size
        self.samples = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                task = json.loads(line)
                for pair in task.get("train_pairs", []):
                    inp = pair["input"]
                    out = pair["output"]
                    h_in, w_in = _grid_size(inp)
                    h_out, w_out = _grid_size(out)
                    if (max(h_in, h_out) <= max_grid_size and
                            max(w_in, w_out) <= max_grid_size):
                        self.samples.append((inp, out))

        if max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, out = self.samples[idx]
        return {
            "type": "arc",
            "input_grid": _pad_grid(inp, self.max_grid_size, self.max_grid_size),
            "target_grid": _pad_grid(out, self.max_grid_size, self.max_grid_size),
            "mask": torch.ones(self.max_grid_size, self.max_grid_size, dtype=torch.long),
        }
