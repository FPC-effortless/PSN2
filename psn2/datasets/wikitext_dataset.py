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
    from psn2.token_ids import stable_token_id
    return stable_token_id(s, vocab)


class WikitextDataset(Dataset):
    """Disabled legacy adapter: full-grid shifted targets reveal future words.

    A replacement requires a registered causal interface and a lossless native
    vocabulary. Old ten-class grid scores are historical diagnostics only.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "The legacy full-grid shifted-target adapter is noncausal and is disabled. "
            "Use an explicitly registered causal language adapter before training."
        )
