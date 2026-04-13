"""Quick verification script to test critical fixes."""
import torch
from psn2.core import PSN2System
from psn2.phases import PhaseController
from psn2.vsa import bind, unbind
from psn2.bonds import BondSystem

print("=" * 60)
print("Verifying Critical Fixes")
print("=" * 60)

# Test 1: PhaseController regime propagation
print("\n[Test 1] PhaseController regime propagation")
model = PSN2System(dim=512, max_nodes=256, grid_vocab=10, rel_vocab=64, stage="D1")
controller = PhaseController(model.node_bank, budget=10, bond_system=model.bond_system)
print(f"  Initial regime: {controller.active_regime}")
dummy_input = torch.randn(512)
controller.run_pulse(dummy_input, regime="compositional")
print(f"  After run_pulse(regime='compositional'): {controller.active_regime}")
assert controller.active_regime == "compositional", "Regime not updated!"
print("  ✅ PASS: Regime propagation works")

# Test 2: VSA circular convolution bind/unbind
print("\n[Test 2] VSA circular convolution bind/unbind")
a = torch.randn(512)
b = torch.randn(512)
a = a / a.norm()
b = b / b.norm()

# Bind and unbind
c = bind(a, b, method="circular")
a_recovered = unbind(c, b, method="circular")

# Check recovery accuracy
similarity = torch.nn.functional.cosine_similarity(a.unsqueeze(0), a_recovered.unsqueeze(0))
print(f"  Original vs recovered similarity: {similarity.item():.4f}")
# Note: Circular convolution has some noise, 0.70+ is acceptable for random vectors
assert similarity.item() > 0.65, f"Recovery failed: {similarity.item()}"
print("  ✅ PASS: Circular convolution bind/unbind works (similarity > 0.65)")

# Test 3: Bond formation in compositional regime
print("\n[Test 3] Bond formation in compositional regime")
bond_system = BondSystem(dim=512)
src_vec = torch.randn(512)
tgt_vec = torch.randn(512)
bond = bond_system.form_bond("causal", 0, 1, src_vec, tgt_vec, shape_type="compositional")
assert bond is not None, "Bond formation failed!"
print(f"  Bond formed: type={bond.bond_type}, strength={bond.strength}")
print("  ✅ PASS: Bond formation works")

# Test 4: Bond recovery
print("\n[Test 4] Bond recovery via unbind + cleanup")
# Create a simple codebook
codebook = torch.stack([src_vec / src_vec.norm(), torch.randn(512)])
codebook[1] = codebook[1] / codebook[1].norm()

idx, recovered_vec, sim = bond_system.recover_source(bond, tgt_vec, codebook)
print(f"  Recovered index: {idx}, similarity: {sim:.4f}")
assert idx == 0, f"Wrong index recovered: {idx}"
assert sim > 0.70, f"Low similarity: {sim}"
print("  ✅ PASS: Bond recovery works")

# Test 5: Prune score uses tau not e
print("\n[Test 5] Prune score logic")
from psn2.growth import GrowthLedger
ledger = GrowthLedger()

# Initialize growth_state if it doesn't exist
if not hasattr(model.node_bank, 'growth_state'):
    model.node_bank.growth_state = torch.zeros(model.max_nodes, 4)

# Create a node with high tau (active) and high e (learning)
model.node_bank.active[0] = 1.0
model.node_bank.tau[0] = 8.0  # Very active
model.node_bank.e[0] = 1.5    # High error (learning)
model.node_bank.growth_state[0, 3] = 50.0  # Age = 50 (young)

score = ledger.compute_prune_score(model.node_bank, 0)
print(f"  Active learning node (tau=8.0, e=1.5, age=50): prune_score={score:.4f}")
assert score < 0.3, f"Active node has high prune score: {score}"

# Create a silent node with low tau and low e
model.node_bank.active[1] = 1.0
model.node_bank.tau[1] = 0.1  # Silent
model.node_bank.e[1] = 0.05   # Low error
model.node_bank.growth_state[1, 3] = 250.0  # Age = 250 (old)

score2 = ledger.compute_prune_score(model.node_bank, 1)
print(f"  Silent node (tau=0.1, e=0.05, age=250): prune_score={score2:.4f}")
assert score2 > score, f"Silent node should have higher prune score than active: {score2} vs {score}"
print("  ✅ PASS: Prune score logic correct")

print("\n" + "=" * 60)
print("All verification tests passed! ✅")
print("=" * 60)
print("\nReady to run training with fixes applied.")
print("Next step: python train_sequential.py --config configs/default.json \\")
print("           --checkpoint-dir /kaggle/working/artifacts \\")
print("           --start-stage D1 --end-stage D1")
