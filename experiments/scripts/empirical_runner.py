#!/usr/bin/env python3
"""
Runner for SYNAPSE empirical paper experiments.

Usage
-----
::

    python experiments/scripts/empirical_runner.py all -v
    python experiments/scripts/empirical_runner.py event_sparse_recovery --verbose
    python experiments/scripts/empirical_runner.py all --config=experiments/configs/empirical.yaml
"""

import argparse
import importlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.plotting import plot_experiment_summary
from experiments.common.report import ExperimentReport, print_report, print_summary, save_report_json
from experiments.common.theme import apply_theme
from experiments.empirical.common.config import load_config, validate_config, to_plain_dict
from experiments.utils.model_io import create_run_capsule, save_config_snapshot, save_metrics, save_run_pointer

# Registry: short name -> module path
EXPERIMENTS = {
    # Foundation: core mathematical verification (PC-01 .. PC-04)
    "event_sparse_recovery": "experiments.empirical.src.foundation.event_sparse_recovery",
    "memory_sufficiency":    "experiments.empirical.src.foundation.memory_sufficiency",
    "topology_value_probe":  "experiments.empirical.src.foundation.topology_value_probe",
    "compression_robustness":"experiments.empirical.src.foundation.compression_robustness",
    # Benchmarks: public data & ablations (PC-05, PC-06)
    "public_benchmark":      "experiments.empirical.src.benchmarks.public_benchmark",
    "ablation_study":        "experiments.empirical.src.benchmarks.ablation_study",
    # Pilot: internal diagnostics (PC-07)
    "pilot_diagnostics":     "experiments.empirical.src.pilot.pilot_diagnostics",
    # Robotics: real-trajectory applied validation (AP-00 .. AP-04)
    "dataset_audit":              "experiments.empirical.src.robotics.dataset_audit",
    "anchor_phase_alignment":     "experiments.empirical.src.robotics.anchor_phase_alignment",
    "compression_retention":      "experiments.empirical.src.robotics.compression_retention",
    "topology_structure":         "experiments.empirical.src.robotics.topology_structure",
    "stability_sensitivity":      "experiments.empirical.src.robotics.stability_sensitivity",
}

# Per-experiment config file mapping
CONFIG_MAP = {
    "event_sparse_recovery": "experiments/configs/foundation.yaml",
    "memory_sufficiency":    "experiments/configs/foundation.yaml",
    "topology_value_probe":  "experiments/configs/foundation.yaml",
    "compression_robustness":"experiments/configs/foundation.yaml",
    "public_benchmark":      "experiments/configs/benchmarks.yaml",
    "ablation_study":        "experiments/configs/benchmarks.yaml",
    "pilot_diagnostics":     "experiments/configs/pilot.yaml",
    "dataset_audit":         "experiments/configs/robotics.yaml",
    "anchor_phase_alignment":"experiments/configs/robotics.yaml",
    "compression_retention": "experiments/configs/robotics.yaml",
    "topology_structure":    "experiments/configs/robotics.yaml",
    "stability_sensitivity": "experiments/configs/robotics.yaml",
}

ORDER = list(EXPERIMENTS.keys())

log = logging.getLogger(__name__)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(sep=" ", timespec="seconds")


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def _import_run_fn(name: str):
    return importlib.import_module(EXPERIMENTS[name]).run_experiment


def run_single(name: str, cfg, verbose: bool) -> ExperimentReport:
    print(f"\n{'-' * 62}")
    print(f"  > Experiment: {name}")
    print(f"  Start: {_timestamp()}")
    print(f"{'-' * 62}")

    start = perf_counter()
    try:
        report = _import_run_fn(name)(cfg=cfg, verbose=verbose)
        print_report(report)
        return report
    except Exception as exc:
        log.error("Experiment %s crashed: %s", name, exc, exc_info=True)
        report = ExperimentReport(
            experiment_id=name.upper(),
            experiment_name=name,
            formal_reference="",
            claim="",
            status="ERROR",
        )
        report.metadata["error"] = str(exc)
        return report
    finally:
        elapsed = perf_counter() - start
        print(f"\n  [Experiment '{name}' Finished]")
        print(f"  End:      {_timestamp()}")
        print(f"  Duration: {_format_elapsed(elapsed)}")


def run_all(verbose: bool, strict: bool) -> bool:
    print()
    print("=" * 62)
    print(" SYNAPSE EMPIRICAL EXPERIMENT SUITE ".center(62))
    print(" experiments/empirical/ ".center(62))
    print("=" * 62)
    print(f"  Started: {_timestamp()}")
    print()

    total_start = perf_counter()
    reports = []
    # Use foundation config for suite-level capsule (arbitrary choice)
    suite_cfg = load_config("experiments/configs/foundation.yaml")
    validate_config(suite_cfg)
    capsule = create_run_capsule(suite_cfg.output_dir, "empirical_suite")

    for name in ORDER:
        exp_config = CONFIG_MAP.get(name, "experiments/configs/foundation.yaml")
        cfg = load_config(exp_config)
        validate_config(cfg)
        report = run_single(name, cfg, verbose)
        reports.append(report)
        save_metrics(capsule, {
            "experiment": name,
            "status": report.status,
            "passed": report.passed_cases,
            "total": report.total_cases,
            "duration": round(report.duration_seconds, 3),
        })
        if strict and report.status != "PASS":
            print(f"\n  [STOP] STRICT MODE: Halting after {name} failure.")
            break

    elapsed = perf_counter() - total_start
    print(f"\n  Total duration: {_format_elapsed(elapsed)}")

    apply_theme(suite_cfg.plotting.theme)
    try:
        plot_experiment_summary(
            experiment_names=[r.experiment_name for r in reports],
            statuses=[r.status for r in reports],
            durations=[r.duration_seconds for r in reports],
            case_counts=[(r.passed_cases, r.total_cases) for r in reports],
            save_path=capsule.figures / "empirical_suite_summary",
            formats=suite_cfg.plotting.formats,
            theme=suite_cfg.plotting.theme,
        )
    except Exception as e:
        log.warning("Suite summary figure failed: %s", e)

    passed = print_summary(reports)

    suite = ExperimentReport(
        experiment_id="EMPIRICAL_SUITE",
        experiment_name="Empirical Paper Experiments",
        formal_reference="experiments/empirical/*",
        claim="All empirical paper claims verified",
        status="PASS" if passed else "FAIL",
        total_cases=sum(r.total_cases for r in reports),
        passed_cases=sum(r.passed_cases for r in reports),
        failed_cases=sum(r.failed_cases for r in reports),
        duration_seconds=elapsed,
    )
    save_report_json(suite, str(capsule.artifacts / "suite_report.json"))
    save_run_pointer(capsule, cfg.output_dir)
    return passed


def print_usage():
    print("\nUsage: python experiments/scripts/empirical_runner.py <experiment> [options]")
    print("\nAvailable experiments:")
    for name in ORDER:
        print(f"  {name:<28} -> {EXPERIMENTS[name]}")
    print(f"  {'all':<28} -> Run all experiments in order")
    print("\nOptions:")
    print("  --verbose, -v            Verbose output")
    print("  --strict                 Stop on first failure")
    print(f"  --config=PATH            YAML config override (auto-selected per experiment)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SYNAPSE Empirical Experiment Runner",
        add_help=False,
    )
    parser.add_argument("experiment", nargs="?", help="Experiment name or 'all'")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--config", default="")
    args, _ = parser.parse_known_args()

    if not args.experiment or args.experiment in ["-h", "--help", "help"]:
        print_usage()
        return 0 if args.experiment in ["-h", "--help", "help"] else 1

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    os.chdir(PROJECT_ROOT)

    experiment = args.experiment.lower().replace("-", "_")

    if experiment == "all":
        return 0 if run_all(args.verbose, args.strict) else 1

    if experiment not in EXPERIMENTS:
        print(f"Error: Unknown experiment '{experiment}'")
        print(f"Valid: {', '.join(ORDER)} or 'all'")
        return 1

    # Load experiment-specific config (or user override)
    config_path = args.config if args.config else CONFIG_MAP.get(experiment, "experiments/configs/foundation.yaml")
    cfg = load_config(config_path)
    validate_config(cfg)

    report = run_single(experiment, cfg, args.verbose)
    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
