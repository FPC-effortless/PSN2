"""Debug script to test PropertyAwareEntityEncoder"""
import torch
import torch.nn as nn
from psn2.core import PropertyAwareEntityEncoder

# Test the encoder
torch.manual_seed(42)
encoder = PropertyAwareEntityEncoder(dim=256, entity_vocab=17)

# Test with some entity IDs
entity_ids = torch.tensor([0, 1, 4, 5, 8, 9, 12, 13, 16])  # Various entities + mask
print(f"Entity IDs: {entity_ids}")

# Decompose to see properties
for eid in entity_ids:
    if eid < 16:
        color = eid // 8
        shape = (eid % 8) // 4
        size = eid % 4
        print(f"Entity {eid:2d}: color={color}, shape={shape}, size={size}")
    else:
        print(f"Entity {eid:2d}: MASK TOKEN")

# Encode
embeddings = encoder(entity_ids)
print(f"\nEmbeddings shape: {embeddings.shape}")
print(f"Embeddings norm: {embeddings.norm(dim=-1)}")

# Check if similar entities have similar embeddings
print("\nCosine similarities:")
print(f"Entity 0 vs 8 (same shape/size, diff color): {torch.cosine_similarity(embeddings[0:1], embeddings[4:5], dim=-1).item():.4f}")
print(f"Entity 0 vs 1 (same color/shape, diff size): {torch.cosine_similarity(embeddings[0:1], embeddings[1:2], dim=-1).item():.4f}")
print(f"Entity 0 vs 4 (same color, diff shape/size): {torch.cosine_similarity(embeddings[0:1], embeddings[2:3], dim=-1).item():.4f}")

# Test with batch
batch_ids = torch.randint(0, 17, (4, 6))
print(f"\nBatch shape: {batch_ids.shape}")
batch_emb = encoder(batch_ids)
print(f"Batch embeddings shape: {batch_emb.shape}")
