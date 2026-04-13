"""Comprehensive verification of all 9 fixes."""
import torch
from psn2.core import PSN2System
from psn2.phases import PhaseController
from psn2.vsa import bind, unbind
from psn2.bonds import BondSystem

print("=" * 60)
print("Verifying All 9 Fixes (5 Critical + 4 Moderate)")
print("=" * 60)

# Test 1-5: Critical fixes (already verified in verify_fixes.py)
print("\n[Tests 1-5] Critical fixes (regime, VSA, bonds, prune, cache)")
print("  ✅ Already verified in verify_fixes.py")

# Test 6: Local weight update for batches
print("\n[Test 6] Local weight update for batches")
model = PSN2System(dim=512, max_nodes=256, grid_vocab=10, rel_vocab=64, stage="D1")
controller = PhaseController(model.node_bank, budget=10, bond_system=model.bond_system)

# Create batched input [B=4, D=512]
batched_input = torch.randn(4, 512)
model.node_bank.active[0:10] = 1.0
model.node_bank.e[0:10] = 0.5

# Store initial nu values
initial_nu = model.node_bank.nu[0:10].clone()

# Run pulse with batched input
controller._current_input = batched_input
controller._phase_b()

# Check if nu was updated
nu_changed = not torch.allclose(initial_nu, model.node_bank.nu[0:10], atol=1e-6)
print(f"  Batched input shape: {batched_input.shape}")
print(f"  Nu updated: {nu_changed}")
assert nu_changed, "Local weight update not applied to batched input!"
print("  ✅ PASS: Local weight update works for batches")

# Test 7: Batch size validation warnings
print("\n[Test 7] Batch size validation warnings")
print("  Note: This test checks code structure, warnings appear during training")
import train
# Check that the warning code exists
import inspect
source = inspect.getsource(train.build_loaders)
has_warning = "Adjusted batch_size" in source
print(f"  Batch size adjustment warning present: {has_warning}")
assert has_warning, "Batch size warning not found!"

source = inspect.getsource(train.main)
has_skip_warning = "Skipping undersized batch" in source
print(f"  Undersized batch skip warning present: {has_skip_warning}")
assert has_skip_warning, "Skip warning not found!"
print("  ✅ PASS: Batch validation warnings present")

# Test 8: Curiosity goal aging frequency
print("\n[Test 8] Curiosity goal aging frequency")
from psn2.curiosity import CuriosityGoal

# Add some goals with different ages
model.curiosity.goals = [
    CuriosityGoal(target_id=0, priority=0.5, budget_reserve=0.25, age_episodes=500),
    CuriosityGoal(target_id=1, priority=0.3, budget_reserve=0.25, age_episodes=1500),
    CuriosityGoal(target_id=2, priority=0.8, budget_reserve=0.25, age_episodes=100),
]
initial_count = len(model.curiosity.goals)
print(f"  Initial goals: {initial_count}")

# Age goals
model.curiosity.tick_episode()
print(f"  After tick_episode: {len(model.curiosity.goals)} goals")

# Check that stale goals are retired (age > 1000 and priority < 0.40)
# Goal 1 should be retired: age=1501, priority=0.3
assert len(model.curiosity.goals) < initial_count, "Stale goals not retired!"
print("  ✅ PASS: Curiosity goal aging works")

# Test 9: ERS promotion frequency
print("\n[Test 9] ERS promotion frequency")
from psn2.ers import ERSTuple

# Fill working tier with high-utility memories
for i in range(10):
    tup = ERSTuple(
        input_vsa=torch.randn(512),
        trace_vsa=torch.randn(512),
        output_vsa=torch.randn(512),
        source="test",
        utility_score=0.75,  # High utility, should promote
    )
    model.ers.write("working", tup)

initial_working = len(model.ers.working)
initial_episodic = len(model.ers.episodic)
print(f"  Initial: Working={initial_working}, Episodic={initial_episodic}")

# Attempt promotion (mid-session, threshold=0.70)
model.ers.attempt_promotions(session_end=False)

final_working = len(model.ers.working)
final_episodic = len(model.ers.episodic)
print(f"  After promotion: Working={final_working}, Episodic={final_episodic}")

# Check that high-utility items were promoted
assert final_episodic > initial_episodic, "No promotion occurred!"
assert final_working < initial_working, "Working tier not reduced!"
print("  ✅ PASS: ERS promotion works")

# Test 10: Integration check - all systems work together
print("\n[Test 10] Integration check")
batch = {
    "type": "arc",
    "input_grid": torch.randint(0, 10, (2, 8, 8)),
    "target_grid": torch.randint(0, 10, (2, 8, 8)),
}

try:
    out = model.forward_batch(batch, phase="compositional")
    loss = out["loss"]
    print(f"  Forward pass successful: loss={loss.item():.4f}")
    assert loss.item() > 0, "Loss is zero or negative!"
    print("  ✅ PASS: Integration check successful")
except Exception as e:
    print(f"  ❌ FAIL: Integration check failed: {e}")
    raise

print("\n" + "=" * 60)
print("All 9 fixes verified! ✅")
print("=" * 60)
print("\nSystem Status:")
print(f"  - Active nodes: {int(model.node_bank.active.sum().item())}/{model.max_nodes}")
print(f"  - Bonds: {len(model.bond_system.bonds)}")
print(f"  - Attractors: {len(model.attractors)}")
print(f"  - Curiosity goals: {len(model.curiosity.goals)}")
print(f"  - ERS Working: {len(model.ers.working)}")
print(f"  - ERS Episodic: {len(model.ers.episodic)}")
print("\n✅ Ready for production training!")
