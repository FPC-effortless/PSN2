"""Diagnostic script to check if property predictions are working"""
import torch
import torch.nn.functional as F
from psn2.core import PSN2System

def make_simple_batch():
    """Create a batch with a single, deterministic pattern"""
    B = 8
    entities = torch.zeros(B, 4, dtype=torch.long)
    relations = torch.zeros(B, 3, 3, dtype=torch.long)
    
    for b in range(B):
        entities[b, 0] = 0  # color=0, shape=0, size=0
        entities[b, 1] = 16  # MASKED
        entities[b, 2] = 8  # color=1, shape=0, size=0
        entities[b, 3] = 4  # color=0, shape=1, size=0
        
        relations[b, 0] = torch.tensor([0, 0, 1])
        relations[b, 1] = torch.tensor([0, 1, 1])
        relations[b, 2] = torch.tensor([0, 1, 2])
    
    target_entity = torch.ones(B, dtype=torch.long)  # Entity 1: color=0, shape=0, size=1
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

torch.manual_seed(42)
model = PSN2System(dim=256, max_nodes=32, grid_vocab=10, rel_vocab=17, stage="D1")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training and checking property predictions...")
print("Target: Entity 1 (color=0, shape=0, size=1)\n")

for step in range(500):
    batch = make_simple_batch()
    
    optimizer.zero_grad()
    output = model.forward_batch(batch, phase="compositional")
    loss = output["loss"]
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        with torch.no_grad():
            # Get property predictions
            if hasattr(model.entity_decoder, 'get_property_logits'):
                prop_logits = model.entity_decoder.get_property_logits()
                
                pred_colors = prop_logits['color'].argmax(dim=-1)
                pred_shapes = prop_logits['shape'].argmax(dim=-1)
                pred_sizes = prop_logits['size'].argmax(dim=-1)
                
                # Reconstruct entity ID from properties
                reconstructed_entities = pred_colors * 8 + pred_shapes * 4 + pred_sizes
                
                pred_entities = output["pred"].argmax(dim=-1)
                target_entities = batch["target_entity"]
                
                print(f"Step {step:3d} | Loss: {loss.item():.4f}")
                print(f"  Direct predictions:        {pred_entities[:4].tolist()}")
                print(f"  Property-based predictions: {reconstructed_entities[:4].tolist()}")
                print(f"  Targets:                    {target_entities[:4].tolist()}")
                print(f"  Predicted colors: {pred_colors[:4].tolist()} (target: [0,0,0,0])")
                print(f"  Predicted shapes: {pred_shapes[:4].tolist()} (target: [0,0,0,0])")
                print(f"  Predicted sizes:  {pred_sizes[:4].tolist()} (target: [1,1,1,1])")
                print()

print("\nNow testing generalization...")
print("Target: Entity 9 (color=1, shape=0, size=1)\n")

def make_generalization_batch():
    B = 8
    entities = torch.zeros(B, 4, dtype=torch.long)
    relations = torch.zeros(B, 3, 3, dtype=torch.long)
    
    for b in range(B):
        entities[b, 0] = 8  # color=1, shape=0, size=0
        entities[b, 1] = 16  # MASKED
        entities[b, 2] = 0  # color=0, shape=0, size=0
        entities[b, 3] = 12  # color=1, shape=1, size=0
        
        relations[b, 0] = torch.tensor([0, 0, 1])
        relations[b, 1] = torch.tensor([0, 1, 1])
        relations[b, 2] = torch.tensor([0, 1, 2])
    
    target_entity = torch.full((B,), 9, dtype=torch.long)  # Entity 9: color=1, shape=0, size=1
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

with torch.no_grad():
    batch = make_generalization_batch()
    output = model.forward_batch(batch, phase="compositional")
    
    if hasattr(model.entity_decoder, 'get_property_logits'):
        prop_logits = model.entity_decoder.get_property_logits()
        
        pred_colors = prop_logits['color'].argmax(dim=-1)
        pred_shapes = prop_logits['shape'].argmax(dim=-1)
        pred_sizes = prop_logits['size'].argmax(dim=-1)
        
        reconstructed_entities = pred_colors * 8 + pred_shapes * 4 + pred_sizes
        
        pred_entities = output["pred"].argmax(dim=-1)
        target_entities = batch["target_entity"]
        
        print(f"Direct predictions:        {pred_entities[:4].tolist()}")
        print(f"Property-based predictions: {reconstructed_entities[:4].tolist()}")
        print(f"Targets:                    {target_entities[:4].tolist()}")
        print(f"Predicted colors: {pred_colors[:4].tolist()} (target: [1,1,1,1])")
        print(f"Predicted shapes: {pred_shapes[:4].tolist()} (target: [0,0,0,0])")
        print(f"Predicted sizes:  {pred_sizes[:4].tolist()} (target: [1,1,1,1])")
        
        # Check if property-based prediction is correct
        prop_correct = (reconstructed_entities == target_entities).sum().item()
        direct_correct = (pred_entities == target_entities).sum().item()
        
        print(f"\nProperty-based accuracy: {prop_correct}/8 = {prop_correct/8:.2f}")
        print(f"Direct prediction accuracy: {direct_correct}/8 = {direct_correct/8:.2f}")
        
        if prop_correct > direct_correct:
            print("\n✓ Property-based predictions are better! The model learned properties.")
        elif prop_correct > 0:
            print("\n~ Property-based predictions show some generalization.")
        else:
            print("\n✗ Property-based predictions also failed to generalize.")
