"""
Experiment 03 -- Exact Change-Point Identification
====================================================

Formal claim
    Theorem 8.3  (docs/formal_math/02_rigorous_architecture.md)
    Under piecewise-constant trajectories with score separation and
    spacing, I*(x_{1:T}) = {c_1, ..., c_m} exactly.

Outputs
    figures/changepoint_sample.{png,pdf}
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

from synapse_core.event_encoder import sharp_event_score
from synapse_core.anchor_selector import select_anchors
from experiments.utils.config import load_config, validate, ExperimentConfig, get_experiment_overrides
from experiments.utils.model_io import create_run_capsule, save_config_snapshot, save_run_pointer
from experiments.utils.logging import setup_run_logging
from experiments.utils.exporter import append_metrics_jsonl, save_results_csv
from experiments.utils.progress import iter_progress
from experiments.common.trajectory_generators import piecewise_constant
from experiments.common.report import (
    ExperimentReport, TestCase, ExperimentTimer, print_report, save_report_json,
)
from experiments.common.theme import apply_theme
from experiments.common.plotting import plot_changepoint_identification

log = logging.getLogger(__name__)

EXPERIMENT_ID = "EXP-03"
EXPERIMENT_NAME = "Exact Change-Point Identification"
FORMAL_REF = "Theorem 8.3 (02_rigorous_architecture.md)"
CLAIM = "I*(x_{1:T}) = {c_1,...,c_m} under separation + spacing"

CSV_FIELDS = [
    "d", "K", "r", "tau", "m_target", "spacing_mode",
    "num_trials", "passed", "fail_trial",
]


def run_experiment(
    cfg: ExperimentConfig | None = None,
    verbose: bool = False,
) -> ExperimentReport:
    if cfg is None:
        cfg = load_config("experiments/configs/default.yaml")
        validate(cfg)

    ov = get_experiment_overrides(cfg, "exp03_changepoint_identification")
    dims           = ov.get("dims", [1, 3, 10])
    num_trials     = ov.get("num_trials", 200)
    jump_mag_factor = ov.get("jump_magnitude_factor", 20.0)

    capsule = create_run_capsule(cfg.output_dir, "changepoint_identification")
    save_config_snapshot(capsule, cfg)
    apply_theme(cfg.plotting.theme)
    setup_run_logging(capsule, verbose=verbose, experiment_id=EXPERIMENT_ID)

    log.info("[%s] %s", EXPERIMENT_ID, EXPERIMENT_NAME)
    log.info("  dims=%s  trials=%d  jump_factor=%.1f", dims, num_trials, jump_mag_factor)

    rng = np.random.default_rng(cfg.execution.seed)

    report = ExperimentReport(
        experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
        formal_reference=FORMAL_REF, claim=CLAIM,
    )
    csv_rows: list[dict] = []
    sample_fig_done = False

    param_configs = [
        # (K, r, tau, m_target, spacing_mode)
        (10, 2, 0.5, 1, "comfortable"), (10, 2, 0.5, 2, "comfortable"),
        (10, 2, 0.5, 5, "comfortable"), (10, 2, 0.5, 10, "comfortable"),
        (10, 2, 0.5, 5, "tight"),       (5, 5, 0.5, 3, "comfortable"),
        (50, 0, 0.5, 10, "comfortable"),
    ]

    with ExperimentTimer() as timer:
        sweep = [(d, K, r, tau, m, sp)
                 for d in dims for K, r, tau, m, sp in param_configs]

        for d, K, r, tau, m_target, spacing_mode in iter_progress(sweep, desc="changepoint"):
            spacing = r + 1 if spacing_mode == "tight" else max(r + 1, r + 5)
            all_exact = True
            error_msg = None
            fail_trial = -1

            for trial in range(num_trials):
                seed_t = rng.integers(0, 2**31)
                m = min(m_target, K)
                change_points = [1 + i * spacing for i in range(m)]
                T = max(change_points[-1] + spacing + 5, 50) if change_points else 50

                traj, _ = piecewise_constant(
                    d, T, change_points,
                    jump_magnitude=tau * jump_mag_factor, seed=seed_t,
                )
                scores = sharp_event_score(traj)

                # Verify assumptions hold
                cp_ok = all(scores[cp] >= tau for cp in change_points)
                non_cp_ok = all(
                    scores[t] < tau for t in range(1, T) if t not in change_points
                )
                if not (cp_ok and non_cp_ok):
                    continue  # assumption violation -- skip

                indices, _ = select_anchors(scores, traj, K, r, tau)
                if indices != change_points:
                    all_exact = False
                    fail_trial = trial
                    error_msg = f"Trial {trial}: I*={indices} != C={change_points}"
                    break

                # Save one sample figure
                if not sample_fig_done and d == dims[0] and m_target >= 2:
                    plot_changepoint_identification(
                        scores, change_points, indices, tau,
                        title=f"{EXPERIMENT_ID}: d={d}, m={m}",
                        save_path=capsule.figures / "changepoint_sample",
                        formats=cfg.plotting.formats, theme=cfg.plotting.theme,
                    )
                    sample_fig_done = True
                    log.info("Figure saved -> %s/changepoint_sample", capsule.figures)

            case_name = f"d{d}_K{K}_r{r}_m{m_target}_{spacing_mode}"
            report.add_case(TestCase(
                name=case_name, passed=all_exact,
                details={"d": d, "K": K, "r": r, "tau": tau,
                         "m_target": m_target, "spacing": spacing_mode,
                         "num_trials": num_trials},
                error=error_msg,
            ))

            row = {"d": d, "K": K, "r": r, "tau": tau, "m_target": m_target,
                   "spacing_mode": spacing_mode, "num_trials": num_trials,
                   "passed": all_exact, "fail_trial": fail_trial}
            csv_rows.append(row)
            append_metrics_jsonl(
                {"case": case_name, **row}, capsule.metrics / "metrics.jsonl",
            )

            if verbose:
                icon = "PASS" if all_exact else "FAIL"
                print(f"  [{icon}] {case_name}")

        # ---- boundary tests ------------------------------------------------
        d_b, T_b, K_b, r_b, tau_b = 3, 100, 10, 2, 1.0
        change_points = [10, 20, 30]
        values = [np.zeros(d_b)]
        for _ in change_points:
            direction = np.zeros(d_b)
            direction[0] = 1.0
            values.append(values[-1].copy() + direction * tau_b)
        traj, _ = piecewise_constant(d_b, T_b, change_points, values=values)
        scores = sharp_event_score(traj)
        indices, _ = select_anchors(scores, traj, K_b, r_b, tau_b)
        passed = indices == change_points
        report.add_case(TestCase(
            name="boundary_scores_at_tau", passed=passed,
            details={"I_star": indices, "expected": change_points},
            error=None if passed else f"I*={indices} != C={change_points}",
        ))

        # Over-budget test
        K_s, r_s, tau_s, d_s = 3, 2, 0.5, 3
        change_points_ob = [5, 12, 19, 26, 33, 40]
        T_ob = 50
        traj, _ = piecewise_constant(d_s, T_ob, change_points_ob,
                                     jump_magnitude=tau_s * 20, seed=999)
        scores = sharp_event_score(traj)
        indices, _ = select_anchors(scores, traj, K_s, r_s, tau_s)
        passed = len(indices) == K_s
        report.add_case(TestCase(
            name="over_budget_m_gt_K", passed=passed,
            details={"K": K_s, "num_cps": len(change_points_ob), "selected": len(indices)},
            error=None if passed else f"|I*|={len(indices)} != K={K_s}",
        ))

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
