"""
Experiment 06 -- Hysteretic Event Encoding
=============================================

Formal claim
    Section 3 of 02_rigorous_architecture.md
    alpha=0 reduces to sharp; alpha>0 produces history dependence.

Outputs
    metrics/metrics.jsonl, results.csv
    artifacts/report.json
    logs/run.log
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synapse_core.event_encoder import sharp_event_score, hysteretic_event_score
from experiments.utils.config import load_config, validate, ExperimentConfig, get_experiment_overrides
from experiments.utils.model_io import create_run_capsule, save_config_snapshot, save_run_pointer
from experiments.utils.logging import setup_run_logging
from experiments.utils.exporter import append_metrics_jsonl, save_results_csv
from experiments.utils.progress import iter_progress
from experiments.common.report import (
    ExperimentReport, TestCase, ExperimentTimer, print_report, save_report_json,
)
from experiments.common.theme import apply_theme

log = logging.getLogger(__name__)

EXPERIMENT_ID = "EXP-06"
EXPERIMENT_NAME = "Hysteretic Event Encoding"
FORMAL_REF = "Section 3 of 02_rigorous_architecture.md"
CLAIM = "alpha=0 == sharp; alpha>0 => history dependence"

CSV_FIELDS = ["test_name", "alpha", "num_trials", "passed", "detail"]


def run_experiment(
    cfg: ExperimentConfig | None = None,
    verbose: bool = False,
) -> ExperimentReport:
    if cfg is None:
        cfg = load_config("experiments/configs/default.yaml")
        validate(cfg)

    ov = get_experiment_overrides(cfg, "exp06_hysteretic_encoding")
    num_sharp_trials  = ov.get("num_sharp_trials", 1000)
    num_alpha0_trials = ov.get("num_alpha0_trials", 500)
    alpha_values      = ov.get("alpha_values", [0.3, 0.5, 0.7, 0.95])
    num_history_pairs = ov.get("num_history_pairs", 200)
    d = cfg.trajectory.d
    score_atol = cfg.verification.score_match_atol
    hist_thresh = cfg.verification.history_diff_threshold

    capsule = create_run_capsule(cfg.output_dir, "hysteretic_encoding")
    save_config_snapshot(capsule, cfg)
    apply_theme(cfg.plotting.theme)
    setup_run_logging(capsule, verbose=verbose, experiment_id=EXPERIMENT_ID)

    log.info("[%s] %s", EXPERIMENT_ID, EXPERIMENT_NAME)
    log.info("  d=%d  sharp_trials=%d  alpha0_trials=%d  history_pairs=%d",
             d, num_sharp_trials, num_alpha0_trials, num_history_pairs)
    log.info("  alpha_values=%s  atol=%.2e  hist_thresh=%.2e",
             alpha_values, score_atol, hist_thresh)

    rng = np.random.default_rng(cfg.execution.seed)

    report = ExperimentReport(
        experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
        formal_reference=FORMAL_REF, claim=CLAIM,
    )
    csv_rows: list[dict] = []

    with ExperimentTimer() as timer:
        # ---- Test A: alpha=0 must equal sharp score -----------------------
        all_match = True
        error_msg = None
        max_diff = 0.0

        for trial in iter_progress(range(num_sharp_trials), desc="alpha=0 vs sharp"):
            T = rng.integers(10, 200)
            traj = rng.standard_normal((T, d))
            sharp_scores = sharp_event_score(traj)
            hyst_scores, _ = hysteretic_event_score(traj, alpha=0.0)

            diff = float(np.max(np.abs(sharp_scores - hyst_scores)))
            max_diff = max(max_diff, diff)

            if diff > score_atol:
                all_match = False
                error_msg = f"Trial {trial}: max_diff={diff:.2e}"
                break

        case_name = "alpha0_equals_sharp"
        report.add_case(TestCase(
            name=case_name, passed=all_match,
            details={"num_trials": num_sharp_trials, "max_diff": max_diff},
            error=error_msg,
        ))
        csv_rows.append({"test_name": case_name, "alpha": 0.0,
                         "num_trials": num_sharp_trials,
                         "passed": all_match, "detail": f"max_diff={max_diff:.2e}"})
        append_metrics_jsonl(
            {"test": case_name, "passed": all_match, "max_diff": max_diff},
            capsule.metrics / "metrics.jsonl",
        )
        if verbose:
            print(f"  [{'PASS' if all_match else 'FAIL'}] {case_name}")

        # ---- Test B: alpha>0 gives history dependence ---------------------
        for alpha in alpha_values:
            found_diff = False
            max_score_diff = 0.0

            for trial in range(num_history_pairs):
                T_h = 20
                x = rng.standard_normal((T_h, d))
                y = rng.standard_normal((T_h, d))
                y[-2:] = x[-2:].copy()  # identical last two steps

                score_x, _ = hysteretic_event_score(x, alpha=alpha)
                score_y, _ = hysteretic_event_score(y, alpha=alpha)
                diff = abs(score_x[-1] - score_y[-1])
                max_score_diff = max(max_score_diff, diff)

                if diff > hist_thresh:
                    found_diff = True
                    break

            case_name = f"history_dep_alpha{alpha}"
            report.add_case(TestCase(
                name=case_name, passed=found_diff,
                details={"alpha": alpha, "num_pairs": num_history_pairs,
                         "max_score_diff": max_score_diff},
                error=None if found_diff else "No history dependence detected",
            ))
            csv_rows.append({"test_name": case_name, "alpha": alpha,
                             "num_trials": num_history_pairs,
                             "passed": found_diff,
                             "detail": f"max_diff={max_score_diff:.2e}"})
            append_metrics_jsonl(
                {"test": case_name, "alpha": alpha, "passed": found_diff,
                 "max_diff": max_score_diff},
                capsule.metrics / "metrics.jsonl",
            )
            if verbose:
                print(f"  [{'PASS' if found_diff else 'FAIL'}] {case_name}")

        # ---- Test C: alpha=0 history independence -------------------------
        all_independent = True
        error_msg = None

        for trial in iter_progress(range(num_alpha0_trials), desc="alpha=0 independence"):
            T_h = 20
            x = rng.standard_normal((T_h, d))
            y = rng.standard_normal((T_h, d))
            y[-2:] = x[-2:].copy()

            score_x, _ = hysteretic_event_score(x, alpha=0.0)
            score_y, _ = hysteretic_event_score(y, alpha=0.0)

            if abs(score_x[-1] - score_y[-1]) > score_atol:
                all_independent = False
                error_msg = f"Trial {trial}: diff={abs(score_x[-1] - score_y[-1]):.2e}"
                break

        case_name = "alpha0_history_independent"
        report.add_case(TestCase(
            name=case_name, passed=all_independent,
            details={"num_trials": num_alpha0_trials},
            error=error_msg,
        ))
        csv_rows.append({"test_name": case_name, "alpha": 0.0,
                         "num_trials": num_alpha0_trials,
                         "passed": all_independent, "detail": ""})
        append_metrics_jsonl(
            {"test": case_name, "passed": all_independent},
            capsule.metrics / "metrics.jsonl",
        )
        if verbose:
            print(f"  [{'PASS' if all_independent else 'FAIL'}] {case_name}")

    report.duration_seconds = timer.elapsed
    report.finalize()

    save_report_json(report, str(capsule.artifacts / "report.json"))
    save_results_csv(csv_rows, CSV_FIELDS, capsule.metrics / "results.csv")
    save_run_pointer(capsule, cfg.output_dir)
    log.info("[%s] %s  (%d/%d)", EXPERIMENT_ID, report.status,
             report.passed_cases, report.total_cases)
    return report


if __name__ == "__main__":
    config_name = "experiments/configs/default.yaml"
    for arg in sys.argv:
        if arg.startswith("--config="):
            config_name = arg.split("=", 1)[1]
    cfg = load_config(config_name)
    validate(cfg)
    v = "--verbose" in sys.argv or "-v" in sys.argv
    report = run_experiment(cfg=cfg, verbose=v)
    print_report(report)
    sys.exit(0 if report.status == "PASS" else 1)
