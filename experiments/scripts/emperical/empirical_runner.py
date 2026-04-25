#!/usr/bin/env python3
"""
Simple Empirical Experiment Runner.

This script purely invokes the empirical experiments sequentially.
It contains NO logic for saving data, plotting, or aggregating results.
All data-saving logic is handled internally by each script and the `seed_runner`.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

# List of all empirical scripts in order
EMPIRICAL_SCRIPTS = {
    "EMP-01": "experiments/empirical/src/components/event_encoder_training.py",
    "EMP-02": "experiments/empirical/src/components/selector_gradient_flow.py",
    "EMP-03": "experiments/empirical/src/components/lift_training.py",
    "EMP-04": "experiments/empirical/src/components/topology_value.py",
    "EMP-05": "experiments/empirical/src/components/train_deploy_consistency.py",
    "EMP-06": "experiments/empirical/src/components/memory_sufficiency_enhanced.py",
    "EMP-07": "experiments/empirical/src/components/scalability_benchmark.py",
    "EMP-08": "experiments/empirical/src/components/hyperparameter_sensitivity.py",
    "EMP-09": "experiments/empirical/src/components/metric_structure_validation.py",
    "EMP-10": "experiments/empirical/src/components/causality_verification.py",
    "EMP-11": "experiments/empirical/src/components/topological_stability_empirical.py",
    "EMP-12": "experiments/empirical/src/components/exact_recovery_empirical.py",
    "EMP-13": "experiments/empirical/src/components/hard_projection_properties.py",
    "FOUNDATION-01": "experiments/empirical/src/foundation/event_sparse_recovery.py",
    "FOUNDATION-02": "experiments/empirical/src/foundation/memory_sufficiency.py",
    "ROBOTICS-01": "experiments/empirical/src/robotics/anchor_phase_alignment.py",
    "ROBOTICS-02": "experiments/empirical/src/robotics/topology_structure.py",
}

def run_script(script_id: str, script_path: str, stop_on_failure: bool = False) -> bool:
    """Invokes a single script as a subprocess."""
    full_path = PROJECT_ROOT / script_path
    
    if not full_path.exists():
        print(f"[ERROR] Script not found: {full_path}")
        return False
        
    print(f"\n{'='*60}")
    print(f"🚀 Running {script_id}: {full_path.name}")
    print(f"{'='*60}")
    
    # Run the script as a separate process to avoid memory/state leaks
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    
    result = subprocess.run([sys.executable, str(full_path)], env=env)
    
    if result.returncode != 0:
        print(f"\n❌ {script_id} FAILED (Exit Code: {result.returncode})")
        if stop_on_failure:
            print("Stopping due to failure (--strict mode).")
            sys.exit(1)
        return False
    else:
        print(f"\n✅ {script_id} COMPLETED SUCCESSFULLY")
        return True

def main():
    parser = argparse.ArgumentParser(description="Lightweight Empirical Suite Runner")
    parser.add_argument(
        "experiment", 
        nargs="?", 
        default="all", 
        help="Experiment ID (e.g., EMP-01) or 'all' to run everything."
    )
    parser.add_argument(
        "--strict", 
        action="store_true", 
        help="Stop execution immediately if a script fails."
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    
    target = args.experiment.upper()
    
    if target == "ALL":
        print(f"Starting execution of all {len(EMPIRICAL_SCRIPTS)} empirical scripts...")
        failures = []
        for eid, path in EMPIRICAL_SCRIPTS.items():
            success = run_script(eid, path, args.strict)
            if not success:
                failures.append(eid)
                
        print(f"\n{'='*60}")
        if failures:
            print(f"⚠️ SUITE FINISHED WITH {len(failures)} FAILURES: {', '.join(failures)}")
        else:
            print("🎉 SUITE FINISHED SUCCESSFULLY! All scripts passed.")
        print(f"{'='*60}")
        
    else:
        if target not in EMPIRICAL_SCRIPTS:
            print(f"Unknown experiment ID: {target}. Valid options: {', '.join(EMPIRICAL_SCRIPTS.keys())}, or 'all'.")
            sys.exit(1)
            
        run_script(target, EMPIRICAL_SCRIPTS[target], args.strict)

if __name__ == "__main__":
    main()
