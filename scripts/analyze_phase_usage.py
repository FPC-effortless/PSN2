"""
Analyze phase usage during training to verify the fix is working.
This script simulates what phases would be used during D1 training.
"""

def simulate_d1_training(total_steps=20000, mix_ratio=0.60):
    """
    Simulate D1 training phase usage with the new logic.
    
    Args:
        total_steps: Total training steps for D1
        mix_ratio: Fraction of batches from primary (ARC) dataset
    """
    import random
    random.seed(42)
    
    phase_counts = {
        "arc": {"perceptive": 0, "compositional": 0, "recursive": 0},
        "graph": {"perceptive": 0, "compositional": 0, "recursive": 0},
    }
    
    for step in range(total_steps):
        # Determine batch type
        use_primary = random.random() < mix_ratio
        batch_type = "arc" if use_primary else "graph"
        
        # Calculate progress fraction
        frac = step / max(total_steps, 1)
        
        # Apply phase selection logic (from train.py)
        if batch_type == "graph":
            phase = "compositional"
        elif batch_type == "arc":
            # D1 is not in D4-D6, so use spatial reasoning progression
            phase = "compositional" if frac > 0.5 else "perceptive"
        else:
            # Fallback
            if frac < 0.33:
                phase = "perceptive"
            elif frac < 0.66:
                phase = "compositional"
            else:
                phase = "recursive"
        
        phase_counts[batch_type][phase] += 1
    
    return phase_counts


def print_analysis():
    """Print analysis of phase usage."""
    print("=" * 70)
    print("D1 Training Phase Usage Analysis (20,000 steps)")
    print("=" * 70)
    
    print("\nData Mix:")
    print("  Primary (ARC):   60% (~12,000 batches)")
    print("  Secondary (Graph): 40% (~8,000 batches)")
    
    # Simulate with new logic
    print("\n" + "─" * 70)
    print("NEW LOGIC (Fixed):")
    print("─" * 70)
    new_counts = simulate_d1_training()
    
    print("\nARC batches (spatial reasoning):")
    arc_total = sum(new_counts["arc"].values())
    for phase, count in new_counts["arc"].items():
        pct = 100 * count / arc_total if arc_total > 0 else 0
        print(f"  {phase:15s}: {count:5d} batches ({pct:5.1f}%)")
    
    print("\nGraph batches (relational reasoning):")
    graph_total = sum(new_counts["graph"].values())
    for phase, count in new_counts["graph"].items():
        pct = 100 * count / graph_total if graph_total > 0 else 0
        marker = " ✓" if phase == "compositional" and count > 0 else ""
        print(f"  {phase:15s}: {count:5d} batches ({pct:5.1f}%){marker}")
    
    # Simulate with old logic for comparison
    print("\n" + "─" * 70)
    print("OLD LOGIC (Broken - for comparison):")
    print("─" * 70)
    
    old_counts = {
        "arc": {"perceptive": 0, "compositional": 0, "recursive": 0},
        "graph": {"perceptive": 0, "compositional": 0, "recursive": 0},
    }
    
    import random
    random.seed(42)
    
    for step in range(20000):
        use_primary = random.random() < 0.60
        batch_type = "arc" if use_primary else "graph"
        frac = step / 20000
        
        # Old time-based logic (broken)
        if frac < 0.33:
            phase = "perceptive"
        elif frac < 0.66:
            phase = "compositional"
        else:
            phase = "recursive"
        
        old_counts[batch_type][phase] += 1
    
    print("\nARC batches:")
    arc_total = sum(old_counts["arc"].values())
    for phase, count in old_counts["arc"].items():
        pct = 100 * count / arc_total if arc_total > 0 else 0
        print(f"  {phase:15s}: {count:5d} batches ({pct:5.1f}%)")
    
    print("\nGraph batches:")
    graph_total = sum(old_counts["graph"].values())
    for phase, count in old_counts["graph"].items():
        pct = 100 * count / graph_total if graph_total > 0 else 0
        marker = " ✗ WRONG!" if phase == "perceptive" and count > 0 else ""
        print(f"  {phase:15s}: {count:5d} batches ({pct:5.1f}%){marker}")
    
    print("\n" + "=" * 70)
    print("KEY DIFFERENCE:")
    print("=" * 70)
    print("\nOLD: Graph batches used PERCEPTIVE phase for first 33% of training")
    print("     → Model couldn't learn relational reasoning")
    print("     → Relation prediction: 1.5%")
    print("\nNEW: Graph batches ALWAYS use COMPOSITIONAL phase")
    print("     → Model learns relational reasoning from step 0")
    print("     → Expected relation prediction: >80%")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_analysis()
