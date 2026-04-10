"""BIG-Bench Hard dataset — D6 full integration stage data.

27 reasoning tasks. Each sample is a (input, target) pair.
Encoded as graph tasks: input words → entity indices, target → class index.
"""
from __future__ import annotations

import json
import random
from typing import Optional

import torch
from torch.utils.data import Dataset


def _tok_hash(s: str, vocab: int) -> int:
    return (hash(s) & 0x7FFFFFFF) % vocab


class BBHDataset(Dataset):
    """BIG-Bench Hard — all 27 tasks merged."""

    def __init__(self, path: str, vocab_size: int = 64, num_entities: int = 6,
                 max_samples: Optional[int] = None, task_filter: Optional[str] = None):
        self.vocab_size = vocab_size
        self.num_entities = num_entities
        self.samples = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if task_filter and obj.get("task") != task_filter:
                    continue
                target_str = str(obj.get("target", "")).strip()
                # Hash all targets uniformly — avoids boolean 0/1 colliding with
                # other hashed targets in the same vocab space
                target = _tok_hash(target_str, vocab_size)
                self.samples.append({
                    "task": obj.get("task", "bbh"),
                    "input": str(obj.get("input", "")),
                    "target": target,
                })

        if max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        words = s["input"].split()
        entities = [_tok_hash(w, self.vocab_size) for w in words[:self.num_entities]]
        while len(entities) < self.num_entities:
            entities.append(0)

        # Use word-pair hashes for relations, same as other graph datasets
        relations = []
        for i in range(self.num_entities - 1):
            rel = _tok_hash(f"{entities[i]}-{entities[i+1]}", 32)
            relations.append([i, rel, i + 1])

        return {
            "type": "graph",
            "entities": torch.tensor(entities, dtype=torch.long),
            "relations": torch.tensor(relations, dtype=torch.long),
            "target_entity": torch.tensor(s["target"], dtype=torch.long),
            "target_relation": torch.tensor(0, dtype=torch.long),
            "mask_entity": torch.tensor(0, dtype=torch.long),
            "mask_relation": torch.tensor(0, dtype=torch.long),
        }
