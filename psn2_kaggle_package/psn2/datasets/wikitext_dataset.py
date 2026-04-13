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

    # Wikitext needs a much larger vocab than the 10-bucket grid_vocab to avoid
    # extreme hash collisions that reduce the signal to noise. We use a fixed
    # internal vocab of 4096 for hashing, then remap to grid_vocab at output so
    # the tensor values stay within the model's grid_encoder embedding range.
    _INTERNAL_VOCAB = 4096

    def __init__(self, path: str, vocab_size: int = 10, grid_size: int = 8,
                 stride: int = 32, max_samples: Optional[int] = None):
        self.vocab_size = vocab_size          # model embedding range (grid_vocab)
        self.grid_size = grid_size
        self.window = grid_size * grid_size  # tokens per sample

        # Build token list from all non-empty lines.
        # Hash into _INTERNAL_VOCAB first for better word discrimination,
        # then mod down to vocab_size when building tensors.
        tokens = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                text = json.loads(line).get("text", "").strip()
                if not text:
                    continue
                for word in text.split():
                    tokens.append(_tok_hash(word, self._INTERNAL_VOCAB))

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
        # Remap internal vocab tokens to model's grid_vocab range
        inp_tok = [t % self.vocab_size for t in window[:self.window]]
        tgt_tok = [t % self.vocab_size for t in window[1:self.window + 1]]
        inp = torch.tensor(inp_tok, dtype=torch.long).reshape(
            self.grid_size, self.grid_size)
        tgt = torch.tensor(tgt_tok, dtype=torch.long).reshape(
            self.grid_size, self.grid_size)
        return {
            "type": "arc",
            "input_grid": inp,
            "target_grid": tgt,
            "mask": torch.ones(self.grid_size, self.grid_size, dtype=torch.long),
        }
