"""Diagnostic script to understand why property-aware encoder isn't helping"""
import torch
import torch.nn.functional as F
from psn2.core import PSN2System, PropertyAwareEntityEncoder

# Create a simple test case
torch.manual_seed(42)

# Test 1: Can the encoder distinguish entities with different properties?
print("="*70)
print("Test 1: Property Embedding Similarity")
print("="*70)

encoder = PropertyAwareEntityEncoder(dim=256, entity_vocab=17)

# Entities with different property combinations
entity_ids = torch.tensor([
    0,   # color=0, shape=0, size=0
    1,   # color=0, shape=0, size=1 (same color/shape, diff size)
    4,   # color=0, shape=1, size=0 (same color, diff shape/size)
    8,   # color=1, shape=0, size=0 (diff color, same shape/size)
])

embeddings = encoder(entity_ids)

print(f"\nCosine similarities:")
print(f"Entity 0 vs 1 (same color+shape): {F.cosine_similarity(embeddings[0:1], embeddings[1:2], dim=-1).item():.4f}")
print(f"Entity 0 vs 4 (same color only):   {F.cosine_similarity(embeddings[0:1], embeddings[2:3], dim=-1).item():.4f}")
print(f"Entity 0 vs 8 (same shape+size):   {F.cosine_similarity(embeddings[0:1], embeddings[3:4], dim=-1).item():.4f}")

# Test 2: Can the model learn a simple pattern?
print("\n" + "="*70)
print("Test 2: Learning Simple Pattern")
print("="*70)

model = PSN2System(dim=256, max_nodes=32, grid_vocab=10, rel_vocab=17, stage="D1")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Create a deterministic test case:
# Entity 0 (color=0, shape=0, size=0) has same_color relation with Entity 1 (color=0, shape=0, size=1)
# Mask Entity 1, model should predict it from Entity 0 + same_color relation

def make_simple_batch():
    """Create a batch with a single, deterministic pattern"""
    B = 8
    entities = torch.zeros(B, 4, dtype=torch.long)
    relations = torch.zeros(B, 3, 3, dtype=torch.long)
    
    for b in range(B):
        # Always use entities 0 and 1 (same color, same shape, different size)
        entities[b, 0] = 0  # color=0, shape=0, size=0
        entities[b, 1] = 16  # MASKED
        entities[b, 2] = 8  # color=1, shape=0, size=0 (different color)
        entities[b, 3] = 4  # color=0, shape=1, size=0 (different shape)
        
        # Relations:
        # 0 same_color 1 (type 0)
        # 0 same_shape 1 (type 1)
        # 0 same_shape 2 (type 1)
        relations[b, 0] = torch.tensor([0, 0, 1])  # 0 same_color 1
        relations[b, 1] = torch.tensor([0, 1, 1])  # 0 same_shape 1
        relations[b, 2] = torch.tensor([0, 1, 2])  # 0 same_shape 2
    
    target_entity = torch.ones(B, dtype=torch.long)  # Always entity 1
    masked_idx = torch.ones(B, dtype=torch.long)  # Always position 1
    target_relation = torch.zeros(B, dtype=torch.long)  # Most common relation type
    
    return {
        "type": "graph",
        "entities": entities,
        "relations": relations,
        "target_entity": target_entity,
        "target_relation": target_relation,
        "masked_entity_idx": masked_idx,
    }

# Train on this simple pattern
print("\nTraining on deterministic pattern: Entity 0 + same_color → Entity 1")
print("Entity 0: color=0, shape=0, size=0")
print("Entity 1: color=0, shape=0, size=1")
print("Pattern: same color, same shape, size differs by 1\n")

for step in range(500):
    batch = make_simple_batch()
    
    optimizer.zero_grad()
    output = model.forward_batch(batch, phase="compositional")
    loss = output["loss"]
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        with torch.no_grad():
            pred_entities = output["pred"].argmax(dim=-1)
            target_entities = batch["target_entity"]
            correct = (pred_entities == target_entities).sum().item()
            accuracy = correct / 8
            
            print(f"Step {step:3d} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}")
            if step == 0:
                print(f"  Predictions: {pred_entities[:4].tolist()}")
                print(f"  Targets:     {target_entities[:4].tolist()}")

print("\n" + "="*70)
print("Test 3: Generalization to Similar Pattern")
print("="*70)

# Now test if it can generalize to entities 8 and 9 (same pattern, different color)
def make_generalization_batch():
    """Test generalization: Entity 8 + same_color → Entity 9"""
    B = 8
    entities = torch.zeros(B, 4, dtype=torch.long)
    relations = torch.zeros(B, 3, 3, dtype=torch.long)
    
    for b in range(B):
        # Use entities 8 and 9 (same color=1, same shape=0, different size)
        entities[b, 0] = 8  # color=1, shape=0, size=0
        entities[b, 1] = 16  # MASKED
        entities[b, 2] = 0  # color=0, shape=0, size=0 (different color)
        entities[b, 3] = 12  # color=1, shape=1, size=0 (different shape)
        
        # Same relation pattern as training
        relations[b, 0] = torch.tensor([0, 0, 1])  # 0 same_color 1
        relations[b, 1] = torch.tensor([0, 1, 1])  # 0 same_shape 1
        relations[b, 2] = torch.tensor([0, 1, 2])  # 0 same_shape 2
    
    target_entity = torch.full((B,), 9, dtype=torch.long)  # Entity 9
    masked_idx = torch.ones(B, dtype=torch.long)
    target_relation = torch.zeros(B, dtype=torch.long)
    
    return {
        "type": "graph",
        "entities": entities,
        "relations": relations,
        "target_entity": target_entity,
        "target_relation": target_relation,
        "masked_entity_idx": masked_idx,
    }

print("\nTesting generalization: Entity 8 + same_color → Entity 9")
print("Entity 8: color=1, shape=0, size=0")
print("Entity 9: color=1, shape=0, size=1")
print("Same pattern as training, but different color\n")

with torch.no_grad():
    batch = make_generalization_batch()
    output = model.forward_batch(batch, phase="compositional")
    pred_entities = output["pred"].argmax(dim=-1)
    target_entities = batch["target_entity"]
    correct = (pred_entities == target_entities).sum().item()
    accuracy = correct / 8
    
    print(f"Generalization Accuracy: {accuracy:.4f}")
    print(f"  Predictions: {pred_entities[:4].tolist()}")
    print(f"  Targets:     {target_entities[:4].tolist()}")
    
    if accuracy > 0.5:
        print("\n✓ SUCCESS: Model generalized to new entities with same property pattern!")
    else:
        print("\n✗ FAILURE: Model did not generalize to new entities")
