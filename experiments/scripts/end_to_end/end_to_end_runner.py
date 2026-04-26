#!/usr/bin/env python3
"""
End-to-end suite runner.

Runs the dedicated end-to-end train/evaluate/deploy/ablation scripts in a
controlled sequence and passes artifacts through the run manifest rather than
guessing from the filesystem.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to sys.path to allow importing from experiments
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from omegaconf import OmegaConf

from experiments.end_to_end.runtime import END_TO_END_ROOT, PROJECT_ROOT, build_run_artifacts, load_json


SCRIPT_CONFIG = OmegaConf.load(END_TO_END_ROOT / "config" / "default.yaml").scripts.runner
SCRIPT_ORDER = {
    "E2E-01": "experiments.end_to_end.scripts.train",
    "E2E-02": "experiments.end_to_end.scripts.evaluate",
    "E2E-03": "experiments.end_to_end.scripts.deploy_test",
    "E2E-04": "experiments.end_to_end.scripts.run_ablations",
}


def manifest_for(output_dir: str, seed: int) -> Path:
    artifacts = build_run_artifacts(output_dir, seed, manifest_name=SCRIPT_CONFIG.manifest_name)
    return artifacts["manifest"]


def read_checkpoint_from_manifest(output_dir: str, seed: int) -> str:
    manifest_path = manifest_for(output_dir, seed)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    checkpoint = manifest.get("best_checkpoint")
    if not checkpoint:
        raise RuntimeError(f"Manifest is missing best_checkpoint: {manifest_path}")
    return checkpoint


def run_module(module_name: str, overrides: List[str]) -> int:
    cmd = [sys.executable, "-m", module_name, *overrides]
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the end-to-end experiment suite.")
    parser.add_argument("experiment", nargs="?", default="all", help="E2E-01..E2E-04 or 'all'.")
    parser.add_argument("--strict", action="store_true", help="Stop on first failure.")
    parser.add_argument("--smoke", action="store_true", help="Enable smoke mode.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for the training run.")
    parser.add_argument(
        "--output-dir",
        default="experiments/outputs/end_to_end",
        help="Root output directory for the training run.",
    )
    args = parser.parse_args()

    target = args.experiment.upper()
    if target != "ALL" and target not in SCRIPT_ORDER:
        valid = ", ".join(["ALL", *SCRIPT_ORDER.keys()])
        print(f"Unknown experiment '{target}'. Valid options: {valid}")
        return 1

    selected = list(SCRIPT_ORDER.items()) if target == "ALL" else [(target, SCRIPT_ORDER[target])]
    failures: List[str] = []

    for script_id, module_name in selected:
        overrides = [f"logging.output_dir={Path(args.output_dir).as_posix()}", f"logging.seed={args.seed}"]
        if args.smoke:
            overrides.append("smoke.enabled=true")

        if script_id in {"E2E-02", "E2E-03"}:
            try:
                checkpoint = read_checkpoint_from_manifest(args.output_dir, args.seed)
            except Exception as exc:
                print(f"[FAILED] {script_id}: {exc}")
                failures.append(script_id)
                if args.strict:
                    return 1
                continue
            overrides.append(f"checkpoint={Path(checkpoint).as_posix()}")

        print(f"\n{'=' * 60}")
        print(f"[RUNNING] {script_id}: {module_name}")
        print(f"{'=' * 60}")
        return_code = run_module(module_name, overrides)
        if return_code != 0:
            print(f"[FAILED] {script_id} exited with code {return_code}")
            failures.append(script_id)
            if args.strict:
                return return_code
        else:
            print(f"[SUCCESS] {script_id}")

    if failures:
        print(f"Suite finished with failures: {', '.join(failures)}")
        return 1

    print("Suite finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
