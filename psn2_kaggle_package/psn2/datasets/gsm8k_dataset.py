"""GSM8K math dataset — D5 abstract reasoning stage data.

Each problem is a word problem with a numeric answer. Encoded as a graph:
- entities = hashed words from the problem
- target_entity = numeric answer (mod vocab_size)
- chain_of_thought stored for future SLS trace reconstruction
"""
from __future__ import annotations

import json
import random
import re
from typing import Optional

import torch
from torch.utils.data import Dataset


def _tok_hash(s: str, vocab: int) -> int:
    return (hash(s) & 0x7FFFFFFF) % vocab


def _parse_number(s: str) -> Optional[int]:
    """Extract integer from answer string."""
    s = s.replace(",", "").strip()
    m = re.search(r"-?\d+", s)
    return int(m.group()) if m else None


class GSM8KDataset(Dataset):
    """GSM8K grade-school math word problems."""

    def __init__(self, path: str, vocab_size: int = 64, num_entities: int = 6,
                 max_samples: Optional[int] = None):
        self.vocab_size = vocab_size
        self.num_entities = num_entities
        self.samples = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                answer_num = _parse_number(str(obj.get("target", "")))
                if answer_num is None:
                    continue
                self.samples.append({
                    "input": obj["input"],
                    "target": answer_num % vocab_size,
                    "chain_of_thought": obj.get("chain_of_thought", ""),
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
            "masked_entity_idx": torch.tensor(0, dtype=torch.long),
            "masked_relation_idx": torch.tensor(0, dtype=torch.long),
        }
