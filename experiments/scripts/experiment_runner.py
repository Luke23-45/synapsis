#!/usr/bin/env python3
"""
Cross-platform execution script for SYNAPSE verification experiments.
Adapted from GibbsQ's experiment_runner.py pattern.

Usage
-----
::

    python experiments/scripts/experiment_runner.py causality
    python experiments/scripts/experiment_runner.py causality --verbose
    python experiments/scripts/experiment_runner.py all --config=experiments/configs/default.yaml
    python experiments/scripts/experiment_runner.py bounded_cardinality --verbose --config=experiments/configs/default.yaml
"""

import sys
import os

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import argparse
import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.config import load_config, validate
from experiments.utils.model_io import create_run_capsule, save_config_snapshot, save_metrics, save_run_pointer
from experiments.common.report import ExperimentReport, print_report, print_summary, save_report_json
from experiments.common.plotting import plot_experiment_summary
from experiments.common.theme import apply_theme

# Experiment registry — maps short names to modules
EXPERIMENTS = {
    "causality":                "experiments.verification.causality",
    "bounded_cardinality":      "experiments.verification.bounded_cardinality",
    "changepoint":              "experiments.verification.changepoint_identification",
    "reconstruction":           "experiments.verification.exact_reconstruction",
    "topological":              "experiments.verification.topological_stability",
    "hysteretic":               "experiments.verification.hysteretic_encoding",
    "information_loss":         "experiments.verification.information_loss",
    "end_to_end":               "experiments.verification.end_to_end",
}

# Canonical ordering for "all" runs
EXPERIMENT_ORDER = [
    "causality",
    "bounded_cardinality",
    "changepoint",
    "reconstruction",
    "topological",
    "hysteretic",
    "information_loss",
    "end_to_end",
]


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(sep=" ", timespec="seconds")


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def _import_experiment(name: str):
    """Dynamically import an experiment module and return its run_experiment function."""
    module_path = EXPERIMENTS[name]
    import importlib
    mod = importlib.import_module(module_path)
    return mod.run_experiment


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_single(experiment_name: str, cfg, verbose: bool) -> ExperimentReport:
    """Run a single experiment with full logging."""
    run_fn = _import_experiment(experiment_name)

    start_time = _timestamp()
    start_perf = perf_counter()

    print(f"\n{'-' * 60}")
    print(f"  > Experiment: {experiment_name}")
    print(f"  Start: {start_time}")
    print(f"{'-' * 60}")

    status = "unknown"
    try:
        report = run_fn(cfg=cfg, verbose=verbose)
        status = report.status
        print_report(report)
        return report
    except Exception as e:
        status = f"CRASHED ({type(e).__name__})"
        logging.getLogger(__name__).error(f"Experiment crashed: {e}", exc_info=True)
        report = ExperimentReport(
            experiment_id=experiment_name.upper(),
            experiment_name=experiment_name,
            formal_reference="",
            claim="",
            status="ERROR",
        )
        report.metadata["error"] = str(e)
        return report
    finally:
        elapsed = perf_counter() - start_perf
        end_time = _timestamp()
        print(f"\n  [Experiment '{experiment_name}' Finished]")
        print(f"  Status:   {status}")
        print(f"  End:      {end_time}")
        print(f"  Duration: {_format_elapsed(elapsed)}")


def run_all(cfg, verbose: bool, strict: bool = False):
    """Run all experiments in canonical order."""
    reports = []
    total_start = perf_counter()

    print()
    print("=" * 62)
    print(" SYNAPSE FORMAL VERIFICATION SUITE ".center(62))
    print(" docs/formal_math/ -> experiments/verification/ ".center(62))
    print("=" * 62)
    print(f"  Config:  {cfg.output_dir}")
    print(f"  Theme:   {cfg.plotting.theme}")
    print(f"  Started: {_timestamp()}")
    print()

    # Create suite-level capsule
    capsule = create_run_capsule(cfg.output_dir, "suite_run")
    save_config_snapshot(capsule, cfg)

    for name in EXPERIMENT_ORDER:
        report = run_single(name, cfg, verbose)
        reports.append(report)

        # Log to suite-level metrics
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

    total_elapsed = perf_counter() - total_start
    print(f"\n  Total duration: {_format_elapsed(total_elapsed)}")

    # Generate summary figure
    apply_theme(cfg.plotting.theme)
    try:
        plot_experiment_summary(
            experiment_names=[r.experiment_name for r in reports],
            statuses=[r.status for r in reports],
            durations=[r.duration_seconds for r in reports],
            case_counts=[(r.passed_cases, r.total_cases) for r in reports],
            save_path=capsule.figures / "suite_summary",
            formats=cfg.plotting.formats,
            theme=cfg.plotting.theme,
        )
        print(f"  [PLOT] Summary figure -> {capsule.figures}/")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Summary figure failed: {e}")

    # Master summary
    all_passed = print_summary(reports)

    # Save suite report
    save_report_json(
        ExperimentReport(
            experiment_id="SUITE",
            experiment_name="Full Verification Suite",
            formal_reference="docs/formal_math/*",
            claim="All formal claims verified",
            status="PASS" if all_passed else "FAIL",
            total_cases=sum(r.total_cases for r in reports),
            passed_cases=sum(r.passed_cases for r in reports),
            failed_cases=sum(r.failed_cases for r in reports),
            duration_seconds=total_elapsed,
        ),
        str(capsule.artifacts / "suite_report.json"),
    )

    save_run_pointer(capsule, cfg.output_dir)
    return all_passed


def print_usage():
    print("\nUsage: python experiments/scripts/experiment_runner.py <experiment> [options]")
    print("\nAvailable experiments:")
    for name in EXPERIMENT_ORDER:
        module = EXPERIMENTS[name]
        print(f"  {name:<25} -> {module}")
    print(f"  {'all':<25} -> Run all experiments in order")
    print("\nOptions:")
    print("  --verbose, -v            Print per-case results")
    print("  --strict                 Stop on first failure")
    print("  --config=PATH            YAML config file (default: experiments/configs/default.yaml)")
    print("\nExamples:")
    print("  python experiments/scripts/experiment_runner.py causality --verbose")
    print("  python experiments/scripts/experiment_runner.py all --config=experiments/configs/default.yaml")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="SYNAPSE Experiment Runner",
        add_help=False,
    )
    parser.add_argument("experiment", nargs="?", help="Experiment name or 'all'")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--config", default="experiments/configs/default.yaml")

    args, extra = parser.parse_known_args()

    if not args.experiment or args.experiment in ["-h", "--help", "help"]:
        print_usage()
        return 0 if args.experiment in ["-h", "--help", "help"] else 1

    _setup_logging(args.verbose)

    # Change to project root
    os.chdir(PROJECT_ROOT)

    # Load config
    cfg = load_config(args.config)
    validate(cfg)

    experiment = args.experiment.lower().replace("-", "_")

    if experiment == "all":
        passed = run_all(cfg, args.verbose, args.strict)
        return 0 if passed else 1

    if experiment not in EXPERIMENTS:
        print(f"Error: Unknown experiment '{experiment}'")
        print(f"Valid: {', '.join(EXPERIMENT_ORDER)} or 'all'")
        return 1

    report = run_single(experiment, cfg, args.verbose)
    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
