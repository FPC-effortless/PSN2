"""Debug script to check if the learnable data generation is working correctly."""
import torch

def make_graph_batch(B=2, N_e=4, N_r=3):
    """Create a relational graph batch with LEARNABLE semantic structure."""
    # Generate entity properties
    colors = torch.randint(0, 4, (B, N_e))  # 4 colors
    shapes = torch.randint(0, 4, (B, N_e))  # 4 shapes
    sizes = torch.randint(0, 4, (B, N_e))   # 4 sizes
    
    # Entity ID encodes properties
    entities = colors * 16 + shapes * 4 + sizes  # [B, N_e] in range [0, 63]
    
    # Build deterministic relations
    relations_list = []
    for b in range(B):
        batch_relations = []
        for i in range(N_e):
            for j in range(i+1, N_e):
                # Relation type 0: same_color
                if colors[b, i] == colors[b, j]:
                    batch_relations.append([i, 0, j])
                # Relation type 1: same_shape
                if shapes[b, i] == shapes[b, j]:
                    batch_relations.append([i, 1, j])
                # Relation type 2: larger_than
                if sizes[b, i] > sizes[b, j]:
                    batch_relations.append([i, 2, j])
        
        # Pad or truncate to N_r relations
        while len(batch_relations) < N_r:
            batch_relations.append([0, 0, 0])  # padding
        relations_list.append(batch_relations[:N_r])
    
    relations = torch.tensor(relations_list, dtype=torch.long)
    
    # Pick a random entity index to mask
    masked_entity_idx = torch.randint(0, N_e, (B,))
    
    # Target entity is the actual entity at the masked position
    target_entity = entities[torch.arange(B), masked_entity_idx]
    
    # Mask the entity
    entities_masked = entities.clone()
    entities_masked[torch.arange(B), masked_entity_idx] = 63  # mask token
    
    return {
        "entities": entities,
        "entities_masked": entities_masked,
        "relations": relations,
        "target_entity": target_entity,
        "masked_entity_idx": masked_entity_idx,
        "colors": colors,
        "shapes": shapes,
        "sizes": sizes,
    }

# Generate a batch and inspect it
batch = make_graph_batch(B=2, N_e=4, N_r=5)

print("=" * 70)
print("Sample Batch Analysis")
print("=" * 70)

for b in range(2):
    print(f"\nBatch {b}:")
    print(f"  Colors: {batch['colors'][b].tolist()}")
    print(f"  Shapes: {batch['shapes'][b].tolist()}")
    print(f"  Sizes:  {batch['sizes'][b].tolist()}")
    print(f"  Entity IDs (unmasked): {batch['entities'][b].tolist()}")
    print(f"  Entity IDs (masked):   {batch['entities_masked'][b].tolist()}")
    print(f"  Masked index: {batch['masked_entity_idx'][b].item()}")
    print(f"  Target entity: {batch['target_entity'][b].item()}")
    
    # Decode target entity
    target_id = batch['target_entity'][b].item()
    target_color = target_id // 16
    target_shape = (target_id % 16) // 4
    target_size = target_id % 4
    print(f"  Target properties: color={target_color}, shape={target_shape}, size={target_size}")
    
    print(f"  Relations:")
    for r in range(5):
        rel = batch['relations'][b, r].tolist()
        if rel != [0, 0, 0]:  # skip padding
            src_idx, rel_type, tgt_idx = rel
            src_id = batch['entities'][b, src_idx].item()
            tgt_id = batch['entities'][b, tgt_idx].item()
            rel_name = ["same_color", "same_shape", "larger_than"][rel_type]
            print(f"    Entity {src_idx} (ID={src_id}) --{rel_name}--> Entity {tgt_idx} (ID={tgt_id})")
            
            # Check if masked entity is involved
            if src_idx == batch['masked_entity_idx'][b].item():
                print(f"      ^ Masked entity is SOURCE, neighbor is entity {tgt_idx}")
            elif tgt_idx == batch['masked_entity_idx'][b].item():
                print(f"      ^ Masked entity is TARGET, neighbor is entity {src_idx}")

print("\n" + "=" * 70)
print("Key Insight:")
print("=" * 70)
print("The model should learn to predict the masked entity's properties")
print("from its neighbors' properties and the relation types.")
print("For example:")
print("  - If neighbor has same_color relation, masked entity has same color")
print("  - If neighbor has same_shape relation, masked entity has same shape")
print("  - If neighbor has larger_than relation, masked entity is smaller")
