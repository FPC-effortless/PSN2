from __future__ import annotations

import random
import torch
from torch.utils.data import Dataset

RELATIONS = ["left_of", "right_of", "above", "below", "same_color", "same_shape", "near", "far"]

def _make_graph(vocab_size: int = 64, num_entities: int = 6):
    entities = [random.randrange(vocab_size) for _ in range(num_entities)]
    relations = []
    for _ in range(num_entities - 1):
        a = random.randrange(num_entities)
        b = random.randrange(num_entities)
        r = random.randrange(len(RELATIONS))
        relations.append((a, r, b))

    # masked prediction: predict one entity and one relation from context
    masked_entity_idx = random.randrange(num_entities)
    masked_relation_idx = random.randrange(len(relations))
    target_entity = entities[masked_entity_idx]
    target_relation = relations[masked_relation_idx][1]

    entities_masked = entities[:]
    # Use vocab_size-1 as a dedicated mask token, distinct from any real entity.
    # Using 0 collides with valid entity index 0, making the task unsolvable.
    mask_token = vocab_size - 1
    entities_masked[masked_entity_idx] = mask_token

    relations_masked = list(relations)
    # Use 0 instead of -1 for masked relation type to avoid invalid embedding indices
    relations_masked[masked_relation_idx] = (
        relations_masked[masked_relation_idx][0],
        0,
        relations_masked[masked_relation_idx][2],
    )

    # Clamp to ensure no negative indices reach embedding lookups
    entities_tensor = torch.tensor(entities_masked, dtype=torch.long).clamp(min=0)
    relations_tensor = torch.tensor(relations_masked, dtype=torch.long).clamp(min=0)

    return {
        "entities": entities_tensor,
        "relations": relations_tensor,
        "target_entity": torch.tensor(target_entity, dtype=torch.long),
        "target_relation": torch.tensor(target_relation, dtype=torch.long),
        "masked_entity_idx": torch.tensor(masked_entity_idx, dtype=torch.long),
        "masked_relation_idx": torch.tensor(masked_relation_idx, dtype=torch.long),
        "mask_entity": torch.tensor(masked_entity_idx, dtype=torch.long),
        "mask_relation": torch.tensor(masked_relation_idx, dtype=torch.long),
    }

class RelationalGraphDataset(Dataset):
    def __init__(self, n_samples: int = 5000, vocab_size: int = 64, num_entities: int = 6):
        self.n_samples = n_samples
        self.vocab_size = vocab_size
        self.num_entities = num_entities

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        item = _make_graph(self.vocab_size, self.num_entities)
        item["type"] = "graph"
        return item
