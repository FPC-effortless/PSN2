"""Wikitext dataset — D4 linguistic grounding stage data.

Encodes text as arc-style grid tasks: a window of tokens is presented as
a "grid" (1D sequence reshaped to grid_size x grid_size), and the target
is the next-token prediction for each position. This gives the USL codec
grounded language signal without requiring a separate language model head.

Empty lines (section separators) are skipped.
"""
from __future__ import annotations

import json
import random
from typing import Optional

import torch
from torch.utils.data import Dataset


def _tok_hash(s: str, vocab: int) -> int:
    return (hash(s) & 0x7FFFFFFF) % vocab


class WikitextDataset(Dataset):
    """
    Loads Wikitext JSONL. Each sample is a sliding window of tokens
    encoded as a grid for next-token prediction.
    """

    def __init__(self, path: str, vocab_size: int = 512, grid_size: int = 8,
                 stride: int = 32, max_samples: Optional[int] = None):
        self.vocab_size = vocab_size
        self.grid_size = grid_size
        self.window = grid_size * grid_size  # tokens per sample

        # Build token list from all non-empty lines
        tokens = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                text = json.loads(line).get("text", "").strip()
                if not text:
                    continue
                for word in text.split():
                    tokens.append(_tok_hash(word, vocab_size))

        # Sliding window samples
        self.samples = []
        for start in range(0, len(tokens) - self.window, stride):
            window = tokens[start: start + self.window + 1]
            self.samples.append(window)
            if max_samples and len(self.samples) >= max_samples:
                break

        if max_samples and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window = self.samples[idx]
        inp = torch.tensor(window[:self.window], dtype=torch.long).reshape(
            self.grid_size, self.grid_size)
        tgt = torch.tensor(window[1:self.window + 1], dtype=torch.long).reshape(
            self.grid_size, self.grid_size)
        return {
            "type": "arc",
            "input_grid": inp,
            "target_grid": tgt,
            "mask": torch.ones(self.grid_size, self.grid_size, dtype=torch.long),
        }
