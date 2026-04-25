#!/usr/bin/env python3
"""
Simple Validation/Verification Suite Runner.

This script purely invokes the formal mathematical verification experiments sequentially.
It contains NO logic for saving data, plotting, or aggregating results.
All logic is handled internally by each specific script.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

# List of all validation/verification scripts in logical order
VALIDATION_SCRIPTS = {
    "VER-01": "experiments/verification/src/encoder_saliency.py",
    "VER-02": "experiments/verification/src/relaxed_selector.py",
    "VER-03": "experiments/verification/src/hard_projection.py",
    "VER-04": "experiments/verification/src/anchor_geometry_readout.py",
    "VER-05": "experiments/verification/src/full_operator.py",
    "VER-06": "experiments/verification/src/exact_recovery.py",
    "VER-07": "experiments/verification/src/topological_stability.py",
    "VER-08": "experiments/verification/src/sufficiency_metric.py",
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
    parser = argparse.ArgumentParser(description="Lightweight Validation Suite Runner")
    parser.add_argument(
        "experiment", 
        nargs="?", 
        default="all", 
        help="Experiment ID (e.g., VER-01) or 'all' to run everything."
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
        print(f"Starting execution of all {len(VALIDATION_SCRIPTS)} validation scripts...")
        failures = []
        for eid, path in VALIDATION_SCRIPTS.items():
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
        if target not in VALIDATION_SCRIPTS:
            print(f"Unknown experiment ID: {target}. Valid options: {', '.join(VALIDATION_SCRIPTS.keys())}, or 'all'.")
            sys.exit(1)
            
        run_script(target, VALIDATION_SCRIPTS[target], args.strict)

if __name__ == "__main__":
    main()
