"""Theory-of-Mind datasets — D3 stage data.

Supports two formats:
  ToM  (d3_tom):  story + question → expected_answer (free-form string)
  ToMi (d3_tomi): premise + hypothesis → label (entailment / not_entailment)

Both are encoded as relational graph tasks: entities are token hashes,
relations encode the story structure, target is the answer class.
"""
from __future__ import annotations

import json
import random
from typing import Optional

import torch
from torch.utils.data import Dataset

# Simple deterministic token hash → vocab index
def _tok_hash(s: str, vocab: int) -> int:
    return (hash(s) & 0x7FFFFFFF) % vocab


class ToMDataset(Dataset):
    """
    ToM dataset (d3_tom): story + question → answer.
    Encoded as a graph batch: entities = hashed words, target = answer hash.
    """

    def __init__(self, path: str, vocab_size: int = 64, num_entities: int = 6,
                 max_samples: Optional[int] = None):
        self.vocab_size = vocab_size
        self.num_entities = num_entities
        self.samples = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append({
                    "story": obj.get("infilled_story", obj.get("story_structure", "")),
                    "question": obj["question"],
                    "answer": obj["expected_answer"],
                })

        if max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Encode story words as entity indices
        words = (s["story"] + " " + s["question"]).split()
        entities = [_tok_hash(w, self.vocab_size) for w in words[:self.num_entities]]
        while len(entities) < self.num_entities:
            entities.append(0)

        # Relations: (i, relation_type, i+1) for adjacent word pairs
        relations = []
        for i in range(self.num_entities - 1):
            rel = _tok_hash(f"{entities[i]}-{entities[i+1]}", 32)
            relations.append([i, rel, i + 1])

        target_entity = _tok_hash(s["answer"], self.vocab_size)
        target_relation = _tok_hash(s["question"], 32)

        return {
            "type": "graph",
            "entities": torch.tensor(entities, dtype=torch.long),
            "relations": torch.tensor(relations, dtype=torch.long),
            "target_entity": torch.tensor(target_entity, dtype=torch.long),
            "target_relation": torch.tensor(target_relation, dtype=torch.long),
            "mask_entity": torch.tensor(0, dtype=torch.long),
            "mask_relation": torch.tensor(0, dtype=torch.long),
        }


class ToMiDataset(Dataset):
    """
    ToMi dataset (d3_tomi): premise + hypothesis → entailment label.
    Binary classification: entailment=1, not_entailment=0.
    """

    LABEL_MAP = {"entailment": 1, "not_entailment": 0}

    def __init__(self, path: str, vocab_size: int = 64, num_entities: int = 6,
                 max_samples: Optional[int] = None):
        self.vocab_size = vocab_size
        self.num_entities = num_entities
        self.samples = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append({
                    "premise": obj["premise"],
                    "hypothesis": obj["hypothesis"],
                    "label": self.LABEL_MAP.get(obj["label"], 0),
                })

        if max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        words = (s["premise"] + " " + s["hypothesis"]).split()
        entities = [_tok_hash(w, self.vocab_size) for w in words[:self.num_entities]]
        while len(entities) < self.num_entities:
            entities.append(0)

        relations = []
        for i in range(self.num_entities - 1):
            rel = _tok_hash(f"{entities[i]}-{entities[i+1]}", 32)
            relations.append([i, rel, i + 1])

        return {
            "type": "graph",
            "entities": torch.tensor(entities, dtype=torch.long),
            "relations": torch.tensor(relations, dtype=torch.long),
            "target_entity": torch.tensor(s["label"], dtype=torch.long),
            "target_relation": torch.tensor(0, dtype=torch.long),
            "mask_entity": torch.tensor(0, dtype=torch.long),
            "mask_relation": torch.tensor(0, dtype=torch.long),
        }
