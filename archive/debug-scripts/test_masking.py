import torch
from psn2.core import PSN2System

def make_graph_batch(B=4, N_e=6, N_r=5, entity_vocab=64, relation_vocab=32):
    entities = torch.randint(0, entity_vocab, (B, N_e))
    relations = torch.randint(0, relation_vocab, (B, N_r, 3))
    relations[:, :, 0] = torch.randint(0, N_e, (B, N_r))
    relations[:, :, 2] = torch.randint(0, N_e, (B, N_r))
    masked_entity_idx = torch.randint(0, N_e, (B,))
    target_entity = entities[torch.arange(B), masked_entity_idx]
    target_relation = torch.randint(0, relation_vocab, (B,))
    return {
        'type': 'graph',
        'entities': entities,
        'relations': relations,
        'target_entity': target_entity,
        'target_relation': target_relation,
        'masked_entity_idx': masked_entity_idx,
    }

torch.manual_seed(42)
model = PSN2System(128, 32, 10, 64)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

correct = 0
total = 0
for step in range(5000):
    batch = make_graph_batch()
    optimizer.zero_grad()
    output = model.forward_batch(batch, phase='compositional')
    loss = output['loss']
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        pred = output['pred'].argmax(dim=-1)
        correct += (pred == batch['target_entity']).sum().item()
        total += 4
    
    if (step + 1) % 1000 == 0:
        acc = correct / total
        print(f'Step {step+1}: Acc={acc:.4f}')
        correct = 0
        total = 0
