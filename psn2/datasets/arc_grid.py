from __future__ import annotations

import random
import torch
from torch.utils.data import Dataset

def _make_task(grid_size: int, vocab: int):
    # Create a simple completion task: hidden row/col pattern.
    grid = torch.randint(0, vocab, (grid_size, grid_size), dtype=torch.long)
    target = grid.clone()

    mode = random.choice(["row", "col", "diag", "block"])
    mask = torch.zeros_like(grid, dtype=torch.bool)

    if mode == "row":
        r = random.randrange(grid_size)
        mask[r, :] = True
    elif mode == "col":
        c = random.randrange(grid_size)
        mask[:, c] = True
    elif mode == "diag":
        mask.fill_(False)
        for i in range(grid_size):
            mask[i, i] = True
    else:
        r = random.randrange(grid_size - 1)
        c = random.randrange(grid_size - 1)
        mask[r:r+2, c:c+2] = True

    inp = grid.clone()
    inp[mask] = 0
    return inp, target, mask.long()

class ARCGridDataset(Dataset):
    def __init__(self, n_samples: int = 5000, grid_size: int = 8, vocab: int = 10):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.vocab = vocab

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        inp, target, mask = _make_task(self.grid_size, self.vocab)
        return {
            "type": "arc",
            "input_grid": inp,
            "target_grid": target,
            "mask": mask,
        }
