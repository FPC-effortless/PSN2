from psn2.core import PSN2System
from psn2.datasets import ARCGridDataset, RelationalGraphDataset

print("Smoke test: import ok")
print("ARC sample keys:", ARCGridDataset(1)[0].keys())
print("Graph sample keys:", RelationalGraphDataset(1)[0].keys())

model = PSN2System(dim=128, max_nodes=16, grid_vocab=10, rel_vocab=64)
print("Model initialized:", len(model.attractors.codebook), len(model.curiosity.goals))
