"""
PSN-2 Sequential Stage Training
Automatically trains D1 → D2 → D3 → D4 → D5 → D6 with gate certification between stages.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch

from psn2.config import load_config
from psn2.utils import set_seed, ensure_dir
from psn2.checkpoint import CheckpointManager
from train import main as train_stage
from evaluate import main as evaluate_stage


STAGE_ORDER = ["D1", "D2", "D3", "D4", "D5", "D6"]

STAGE_STEPS = {
    "D1": 20000,
    "D2": 20000,
    "D3": 15000,
    "D4": 25000,
    "D5": 30000,
    "D6": 40000,
}

STAGE_GATES = {
    "D1": ["object_tracking_accuracy", "causal_prediction_error",
           "temporal_trace_persistence", "vsa_binding_accuracy"],
    "D2": ["causal_intervention_accuracy", "abstract_analogy_score",
           "vsa_causal_bond_recall", "compositional_split_score"],
    "D3": ["goal_inference_accuracy", "false_belief_accuracy",
           "trust_calibration_rmse", "emotional_shape_induction_accuracy"],
    "D4": ["usl_roundtrip_fidelity", "language_grounded_analogy",
           "isl_coherent_episodes", "linguistic_bond_vsa_recovery"],
    "D5": ["arc_agi_improvement", "math_verification_rate",
           "multi_step_planning_success", "compositional_split_score"],
    "D6": ["sample_complexity_ratio", "few_shot_k1_efficiency",
           "few_shot_k5_efficiency", "compositional_split_score",
           "growth_ledger_i24", "anti_forgetting_regression",
           "human_parity_profile"],
}


def check_stage_gates(eval_results: dict, stage: str) -> tuple[bool, list[str]]:
    """
    Check if all gates for a stage have passed.
    Returns (all_passed, failed_gates).
    """
    gates = STAGE_GATES.get(stage, [])
    stage_key = f"{stage.lower()}_gates"
    
    if stage_key not in eval_results:
        print(f"  Warning: No gate results found for {stage}")
        return False, gates
    
    gate_results = eval_results[stage_key]
    failed = []
    
    for gate in gates:
        if not gate_results.get(gate, False):
            failed.append(gate)
    
    return len(failed) == 0, failed


def activate_stage_subsystems(model, stage: str):
    """Activate subsystems required for each stage."""
    if stage == "D3" and model.ess is None:
        print(f"  Activating ESS (Emotional Shape System) for {stage}")
        model.activate_ess()
    
    if stage == "D4" and model.usl is None:
        print(f"  Activating USL/ISL (Universal Symbolic Language) for {stage}")
        model.activate_usl(vocab_size=1000)
    
    if stage == "D5" and model.tae_tal is None:
        print(f"  Activating TAE (Temporal Abstraction Engine) for {stage}")
        model.activate_tae()


def train_and_evaluate_stage(stage: str, config_path: str, 
                             checkpoint_dir: str, 
                             resume_from: Optional[str] = None,
                             force_continue: bool = False) -> tuple[bool, dict]:
    """
    Train a single stage and evaluate it.
    Returns (passed, eval_results).
    """
    print(f"\n{'='*70}")
    print(f"STAGE {stage}")
    print(f"{'='*70}\n")
    
    # Load config and update for this stage
    cfg = load_config(config_path)
    cfg["stage"] = stage
    cfg["steps"] = STAGE_STEPS[stage]
    cfg["checkpoint_dir"] = checkpoint_dir
    
    # Save stage-specific config
    stage_config_path = os.path.join(checkpoint_dir, f"config_{stage}.json")
    with open(stage_config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    
    print(f"Stage: {stage}")
    print(f"Steps: {cfg['steps']}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print()
    
    # Train the stage
    print(f"[{stage}] Starting training...")
    sys.argv = ["train.py", "--config", stage_config_path]
    if resume_from:
        sys.argv.extend(["--resume", resume_from])
    
    try:
        train_stage()
    except Exception as e:
        print(f"\n[{stage}] Training failed: {e}")
        if not force_continue:
            raise
        print(f"[{stage}] Continuing despite error (--force-continue enabled)")
    
    # Find the latest checkpoint
    latest_ckpt = os.path.join(checkpoint_dir, "latest.pt")
    if not os.path.exists(latest_ckpt):
        print(f"\n[{stage}] ERROR: No checkpoint found at {latest_ckpt}")
        return False, {}
    
    # Evaluate the stage
    print(f"\n[{stage}] Evaluating...")
    eval_output = os.path.join(checkpoint_dir, f"eval_{stage}.json")
    sys.argv = [
        "evaluate.py",
        "--config", stage_config_path,
        "--checkpoint", latest_ckpt,
        "--output", eval_output,
    ]
    
    try:
        evaluate_stage()
    except Exception as e:
        print(f"\n[{stage}] Evaluation failed: {e}")
        if not force_continue:
            raise
        print(f"[{stage}] Continuing despite error (--force-continue enabled)")
        return False, {}
    
    # Load evaluation results
    if not os.path.exists(eval_output):
        print(f"\n[{stage}] ERROR: No evaluation results at {eval_output}")
        return False, {}
    
    with open(eval_output) as f:
        eval_results = json.load(f)
    
    # Check gates
    print(f"\n[{stage}] Gate Certification:")
    passed, failed_gates = check_stage_gates(eval_results, stage)
    
    if passed:
        print(f"  ✓ All gates PASSED for {stage}")
    else:
        print(f"  ✗ {len(failed_gates)} gate(s) FAILED:")
        for gate in failed_gates:
            print(f"    - {gate}")
    
    # Print key metrics
    print(f"\n[{stage}] Key Metrics:")
    if "grid_accuracy" in eval_results:
        print(f"  Grid accuracy:       {eval_results['grid_accuracy']:.4f}")
    if "relation_prediction" in eval_results:
        print(f"  Relation prediction: {eval_results['relation_prediction']:.4f}")
    if "trace_persistence" in eval_results:
        print(f"  Trace persistence:   {eval_results['trace_persistence']:.4f}")
    
    return passed, eval_results


def main():
    parser = argparse.ArgumentParser(
        description="Train PSN-2 sequentially through all developmental stages"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.json",
        help="Base config file"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_sequential",
        help="Directory for checkpoints and results"
    )
    parser.add_argument(
        "--start-stage",
        type=str,
        default="D1",
        choices=STAGE_ORDER,
        help="Stage to start from (default: D1)"
    )
    parser.add_argument(
        "--end-stage",
        type=str,
        default="D6",
        choices=STAGE_ORDER,
        help="Stage to end at (default: D6)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume from checkpoint (for first stage only)"
    )
    parser.add_argument(
        "--skip-gate-check",
        action="store_true",
        help="Continue to next stage even if gates fail"
    )
    parser.add_argument(
        "--force-continue",
        action="store_true",
        help="Continue even if training/evaluation fails"
    )
    args = parser.parse_args()
    
    # Setup
    ensure_dir(args.checkpoint_dir)
    
    # Determine stage range
    start_idx = STAGE_ORDER.index(args.start_stage)
    end_idx = STAGE_ORDER.index(args.end_stage)
    stages_to_run = STAGE_ORDER[start_idx:end_idx + 1]
    
    print("="*70)
    print("PSN-2 SEQUENTIAL TRAINING")
    print("="*70)
    print(f"\nStages to train: {' → '.join(stages_to_run)}")
    print(f"Total steps: {sum(STAGE_STEPS[s] for s in stages_to_run)}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Skip gate checks: {args.skip_gate_check}")
    print()
    
    # Track results
    all_results = {}
    resume_checkpoint = args.resume
    
    # Train each stage sequentially
    for stage in stages_to_run:
        passed, eval_results = train_and_evaluate_stage(
            stage=stage,
            config_path=args.config,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=resume_checkpoint,
            force_continue=args.force_continue,
        )
        
        all_results[stage] = {
            "passed": passed,
            "eval_results": eval_results,
        }
        
        # Check if we should continue
        if not passed and not args.skip_gate_check:
            print(f"\n{'='*70}")
            print(f"STOPPING: {stage} gates failed and --skip-gate-check not set")
            print(f"{'='*70}\n")
            print("Options:")
            print(f"  1. Continue anyway: --skip-gate-check")
            print(f"  2. Resume from {stage}: --start-stage {stage}")
            print(f"  3. Investigate failures in {args.checkpoint_dir}/eval_{stage}.json")
            break
        
        # Use this stage's checkpoint for next stage
        resume_checkpoint = os.path.join(args.checkpoint_dir, "latest.pt")
        
        # Save intermediate results
        results_path = os.path.join(args.checkpoint_dir, "sequential_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
    
    # Final summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}\n")
    
    print("Stage Summary:")
    for stage in stages_to_run:
        if stage in all_results:
            result = all_results[stage]
            status = "✓ PASSED" if result["passed"] else "✗ FAILED"
            print(f"  {stage}: {status}")
        else:
            print(f"  {stage}: NOT RUN")
    
    print(f"\nResults saved to: {args.checkpoint_dir}/sequential_results.json")
    print(f"Final checkpoint: {args.checkpoint_dir}/latest.pt")
    
    # Check if all stages passed
    all_passed = all(
        all_results.get(s, {}).get("passed", False) 
        for s in stages_to_run
    )
    
    if all_passed:
        print("\n🎉 ALL STAGES CERTIFIED! 🎉")
        print("Your model is ready for deployment.")
    else:
        failed_stages = [
            s for s in stages_to_run 
            if not all_results.get(s, {}).get("passed", False)
        ]
        print(f"\n⚠️  {len(failed_stages)} stage(s) need attention: {', '.join(failed_stages)}")


if __name__ == "__main__":
    main()
