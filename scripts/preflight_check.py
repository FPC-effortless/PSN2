"""
Pre-flight check for PSN-2 training.
Verifies all components are working before starting long training runs.
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_imports():
    """Test all critical imports."""
    print("=" * 70)
    print("CHECKING IMPORTS")
    print("=" * 70)
    
    try:
        from psn2.core import PSN2System
        from psn2.config import load_config
        from psn2.datasets import (
            ARCGridDataset, RelationalGraphDataset,
            ARCAGI2Dataset, ToMDataset, ToMiDataset,
            WikitextDataset, GSM8KDataset, BBHDataset
        )
        from psn2.checkpoint import CheckpointManager
        from psn2.phases import PhaseController
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_datasets():
    """Test synthetic datasets."""
    print("\n" + "=" * 70)
    print("CHECKING DATASETS")
    print("=" * 70)
    
    try:
        from psn2.datasets import ARCGridDataset, RelationalGraphDataset
        
        # Test ARC dataset
        arc = ARCGridDataset(n_samples=5, grid_size=8, vocab=10)
        arc_sample = arc[0]
        assert arc_sample["type"] == "arc", f"Expected type='arc', got '{arc_sample['type']}'"
        assert "input_grid" in arc_sample
        assert "target_grid" in arc_sample
        print(f"✅ ARC dataset: {len(arc)} samples, type='{arc_sample['type']}'")
        
        # Test Graph dataset
        graph = RelationalGraphDataset(n_samples=5, vocab_size=64)
        graph_sample = graph[0]
        assert graph_sample["type"] == "graph", f"Expected type='graph', got '{graph_sample['type']}'"
        assert "entities" in graph_sample
        assert "relations" in graph_sample
        print(f"✅ Graph dataset: {len(graph)} samples, type='{graph_sample['type']}'")
        
        return True
    except Exception as e:
        print(f"❌ Dataset check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_phase_selection():
    """Test phase selection logic."""
    print("\n" + "=" * 70)
    print("CHECKING PHASE SELECTION")
    print("=" * 70)
    
    try:
        # Simulate phase selection logic from train.py
        def get_phase(batch_type, stage, frac):
            if batch_type == "graph":
                return "compositional"
            elif batch_type == "arc":
                if stage in ["D4", "D5", "D6"]:
                    return "recursive" if frac > 0.5 else "compositional"
                else:
                    return "compositional" if frac > 0.5 else "perceptive"
            else:
                if frac < 0.33:
                    return "perceptive"
                elif frac < 0.66:
                    return "compositional"
                else:
                    return "recursive"
        
        # Test D1 with graph batches (the critical fix)
        phase = get_phase("graph", "D1", 0.0)
        assert phase == "compositional", f"D1 graph at 0% should be compositional, got {phase}"
        phase = get_phase("graph", "D1", 0.5)
        assert phase == "compositional", f"D1 graph at 50% should be compositional, got {phase}"
        print("✅ D1 graph batches: Always compositional")
        
        # Test D1 with ARC batches
        phase = get_phase("arc", "D1", 0.0)
        assert phase == "perceptive", f"D1 arc at 0% should be perceptive, got {phase}"
        phase = get_phase("arc", "D1", 0.75)
        assert phase == "compositional", f"D1 arc at 75% should be compositional, got {phase}"
        print("✅ D1 ARC batches: perceptive → compositional")
        
        # Test D4 with ARC batches (language)
        phase = get_phase("arc", "D4", 0.25)
        assert phase == "compositional", f"D4 arc at 25% should be compositional, got {phase}"
        phase = get_phase("arc", "D4", 0.75)
        assert phase == "recursive", f"D4 arc at 75% should be recursive, got {phase}"
        print("✅ D4 ARC batches: compositional → recursive")
        
        return True
    except Exception as e:
        print(f"❌ Phase selection check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_creation():
    """Test model instantiation."""
    print("\n" + "=" * 70)
    print("CHECKING MODEL CREATION")
    print("=" * 70)
    
    try:
        from psn2.core import PSN2System
        
        model = PSN2System(
            dim=512,
            max_nodes=256,
            grid_vocab=10,
            rel_vocab=64,
            stage="D1"
        )
        print(f"✅ Model created: {model.max_nodes} nodes, {model.dim}D VSA")
        
        # Test forward pass with synthetic data
        from psn2.datasets import ARCGridDataset, RelationalGraphDataset
        
        arc_ds = ARCGridDataset(n_samples=2, grid_size=8, vocab=10)
        arc_batch = {k: torch.stack([arc_ds[i][k] for i in range(2)]) 
                     for k in arc_ds[0] if k != "type"}
        arc_batch["type"] = "arc"
        
        out = model.forward_batch(arc_batch, phase="perceptive")
        assert "loss" in out
        assert "loss_pred" in out
        print(f"✅ ARC forward pass: loss={out['loss'].item():.4f}")
        
        graph_ds = RelationalGraphDataset(n_samples=2, vocab_size=64)
        graph_batch = {k: torch.stack([graph_ds[i][k] for i in range(2)]) 
                       for k in graph_ds[0] if k != "type"}
        graph_batch["type"] = "graph"
        
        out = model.forward_batch(graph_batch, phase="compositional")
        assert "loss" in out
        assert "loss_pred" in out
        print(f"✅ Graph forward pass: loss={out['loss'].item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_config():
    """Test config loading."""
    print("\n" + "=" * 70)
    print("CHECKING CONFIGURATION")
    print("=" * 70)
    
    try:
        from psn2.config import load_config
        
        config_path = "configs/default.json"
        if not Path(config_path).exists():
            print(f"⚠️  Config file not found: {config_path}")
            return False
        
        cfg = load_config(config_path)
        required_keys = ["vsa_dim", "max_nodes", "batch_size", "steps", 
                        "grid_vocab", "rel_vocab_size", "checkpoint_dir"]
        
        for key in required_keys:
            if key not in cfg:
                print(f"❌ Missing config key: {key}")
                return False
        
        print(f"✅ Config loaded: {len(cfg)} keys")
        print(f"   VSA dim: {cfg['vsa_dim']}")
        print(f"   Max nodes: {cfg['max_nodes']}")
        print(f"   Batch size: {cfg['batch_size']}")
        
        return True
    except Exception as e:
        print(f"❌ Config check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_cuda():
    """Check CUDA availability."""
    print("\n" + "=" * 70)
    print("CHECKING CUDA")
    print("=" * 70)
    
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"✅ CUDA available: {n_gpus} GPU(s)")
        for i in range(n_gpus):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("⚠️  CUDA not available (will use CPU)")
        return True  # Not a failure, just a warning


def main():
    """Run all pre-flight checks."""
    print("\n" + "=" * 70)
    print("PSN-2 PRE-FLIGHT CHECK")
    print("=" * 70)
    
    checks = [
        ("Imports", check_imports),
        ("Datasets", check_datasets),
        ("Phase Selection", check_phase_selection),
        ("Model Creation", check_model_creation),
        ("Configuration", check_config),
        ("CUDA", check_cuda),
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} check crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nResult: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Ready for training.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} check(s) failed. Fix issues before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
