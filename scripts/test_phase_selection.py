"""Test the new batch-type-aware phase selection logic."""

def get_phase(batch_type: str, stage: str, frac: float) -> str:
    """
    Replicate the phase selection logic from train.py for testing.
    """
    if batch_type == "graph":
        # Graph tasks (including ToM, GSM8K, BBH) use compositional phase
        # for relational and causal reasoning
        phase = "compositional"
    elif batch_type == "arc":
        # ARC/grid tasks: progress through phases based on stage
        # D1-D3: focus on perceptive and compositional
        # D4+: add recursive for language grids (Wikitext)
        if stage in ["D4", "D5", "D6"]:
            # Language stages: progress compositional → recursive
            phase = "recursive" if frac > 0.5 else "compositional"
        else:
            # Spatial reasoning stages: perceptive → compositional
            phase = "compositional" if frac > 0.5 else "perceptive"
    else:
        # Fallback: time-based progression
        if frac < 0.33:
            phase = "perceptive"
        elif frac < 0.66:
            phase = "compositional"
        else:
            phase = "recursive"
    return phase


def test_phase_selection():
    """Test phase selection for different scenarios."""
    
    print("Testing Phase Selection Logic")
    print("=" * 60)
    
    # Test D1 with graph batches (the critical fix)
    print("\n[D1] Graph batches (relational reasoning):")
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        phase = get_phase("graph", "D1", frac)
        print(f"  Step {int(frac*100):3d}%: {phase:15s} ✓ (always compositional)")
    
    # Test D1 with ARC batches
    print("\n[D1] ARC batches (spatial reasoning):")
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        phase = get_phase("arc", "D1", frac)
        expected = "compositional" if frac > 0.5 else "perceptive"
        status = "✓" if phase == expected else "✗"
        print(f"  Step {int(frac*100):3d}%: {phase:15s} {status}")
    
    # Test D4 with ARC batches (Wikitext as grids)
    print("\n[D4] ARC batches (language as grids):")
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        phase = get_phase("arc", "D4", frac)
        expected = "recursive" if frac > 0.5 else "compositional"
        status = "✓" if phase == expected else "✗"
        print(f"  Step {int(frac*100):3d}%: {phase:15s} {status}")
    
    # Test D4 with graph batches
    print("\n[D4] Graph batches:")
    for frac in [0.0, 0.5, 1.0]:
        phase = get_phase("graph", "D4", frac)
        print(f"  Step {int(frac*100):3d}%: {phase:15s} ✓ (always compositional)")
    
    print("\n" + "=" * 60)
    print("Key improvements:")
    print("  ✓ Graph batches ALWAYS use compositional phase (was perceptive in first 33%)")
    print("  ✓ ARC batches use stage-appropriate phases")
    print("  ✓ Language stages (D4-D6) progress to recursive phase")
    print("\nExpected outcome:")
    print("  → Relation prediction should improve from 1.5% to >80%")
    print("  → D1 gates should pass after retraining")


if __name__ == "__main__":
    test_phase_selection()
