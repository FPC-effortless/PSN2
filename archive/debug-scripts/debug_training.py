"""Debug script to test training with PropertyAwareEntityEncoder"""
import torch
import torch.nn.functional as F
from psn2.core import PSN2System

def make_graph_batch(B=4, N_e=6, N_r=5):
    """Create a simple test batch"""
    # Generate entity properties
    colors = torch.randint(0, 2, (B, N_e))
    shapes = torch.randint(0, 2, (B, N_e))
    sizes = torch.randint(0, 4, (B, N_e))
    
    # Entity ID encodes properties
    entities = colors * 8 + shapes * 4 + sizes
    
    # Build deterministic relations
    relations_list = []
    masked_entity_idx_list = []
    
    for b in range(B):
        batch_relations = []
        
        for i in range(N_e):
            for j in range(i+1, N_e):
                if colors[b, i] == colors[b, j]:
                    batch_relations.append([i, 0, j])
                if shapes[b, i] == shapes[b, j]:
                    batch_relations.append([i, 1, j])
                if sizes[b, i] > sizes[b, j]:
                    batch_relations.append([i, 2, j])
        
        # Pick a masked entity with relations
        masked_idx = 0
        for attempt in range(10):
            candidate_idx = torch.randint(0, N_e, (1,)).item()
            has_relation = any(
                (rel[0] == candidate_idx or rel[2] == candidate_idx)
                for rel in batch_relations
            )
            if has_relation:
                masked_idx = candidate_idx
                break
        
        masked_entity_idx_list.append(masked_idx)
        
        while len(batch_relations) < N_r:
            batch_relations.append([0, 0, 0])
        relations_list.append(batch_relations[:N_r])
    
    relations = torch.tensor(relations_list, dtype=torch.long)
    masked_entity_idx = torch.tensor(masked_entity_idx_list, dtype=torch.long)
    
    target_entity = entities[torch.arange(B), masked_entity_idx]
    
    entities_masked = entities.clone()
    entities_masked[torch.arange(B), masked_entity_idx] = 16
    
    target_relation = relations[:, :, 1].mode(dim=1)[0]
    
    return {
        "type": "graph",
        "entities": entities_masked,
        "relations": relations,
        "target_entity": target_entity,
        "target_relation": target_relation,
        "masked_entity_idx": masked_entity_idx,
    }

# Initialize model
torch.manual_seed(42)
model = PSN2System(dim=256, max_nodes=32, grid_vocab=10, rel_vocab=17, stage="D1")
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train for a few steps and inspect
print("Training for 100 steps...")
for step in range(100):
    batch = make_graph_batch(B=4, N_e=6, N_r=5)
    
    optimizer.zero_grad()
    output = model.forward_batch(batch, phase="compositional")
    loss = output["loss"]
    loss.backward()
    
    # Check gradients
    if step % 20 == 0:
        grad_norms = model.log_gradient_norms()
        pred_entities = output["pred"].argmax(dim=-1)
        target_entities = batch["target_entity"]
        correct = (pred_entities == target_entities).sum().item()
        accuracy = correct / 4
        
        print(f"Step {step:3d} | Loss: {loss.item():.4f} | Acc: {accuracy:.4f} | Grads: {grad_norms}")
        
        # Inspect a sample
        if step == 0:
            print(f"\nSample batch:")
            print(f"  Entities (masked): {batch['entities'][0]}")
            print(f"  Target entity: {batch['target_entity'][0]}")
            print(f"  Masked idx: {batch['masked_entity_idx'][0]}")
            print(f"  Prediction: {pred_entities[0]}")
            print(f"  Relations: {batch['relations'][0]}")
    
    optimizer.step()

print("\nFinal test:")
batch = make_graph_batch(B=4, N_e=6, N_r=5)
with torch.no_grad():
    output = model.forward_batch(batch, phase="compositional")
    pred_entities = output["pred"].argmax(dim=-1)
    target_entities = batch["target_entity"]
    correct = (pred_entities == target_entities).sum().item()
    accuracy = correct / 4
    print(f"Final accuracy: {accuracy:.4f}")
