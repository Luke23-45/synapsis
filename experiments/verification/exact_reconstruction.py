"""
Experiment 04 -- Exact Reconstruction
========================================

Formal claim
    Corollary 8.4  (docs/formal_math/02_rigorous_architecture.md)
    Under Theorem 8.3's assumptions, the trajectory can be reconstructed
    exactly from x_1 together with the anchor sequence A(x_{1:T}).

Outputs
    figures/reconstruction_sample.{png,pdf}
    metrics/metrics.jsonl, results.csv
    artifacts/report.json
    logs/run.log
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synapse_core.event_encoder import sharp_event_score
from synapse_core.anchor_selector import select_anchors, Anchor
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
from experiments.common.plotting import plot_reconstruction_error

log = logging.getLogger(__name__)

EXPERIMENT_ID = "EXP-04"
EXPERIMENT_NAME = "Exact Reconstruction"
FORMAL_REF = "Corollary 8.4 (02_rigorous_architecture.md)"
CLAIM = "Trajectory recoverable exactly from x_1 + A(x_{1:T})"

CSV_FIELDS = ["d", "T", "m_target", "num_trials", "valid_trials", "skipped_trials", "max_error", "passed"]


def reconstruct_trajectory(
    x_1: np.ndarray,
    anchors: List[Anchor],
    T: int,
) -> np.ndarray:
    """Reconstruct piecewise-constant trajectory from x_1 and anchor sequence.

    Logic (Corollary 8.4):
        x(t) = x_1 for t < first anchor index
        x(t) = s_j  for t in [anchor_j, anchor_{j+1})
        x(t) = s_m  for t >= last anchor
    """
    d = x_1.shape[0]
    trajectory = np.zeros((T, d), dtype=np.float64)

    if not anchors:
        trajectory[:] = x_1
        return trajectory

    trajectory[:anchors[0].index] = x_1
    for j in range(len(anchors)):
        start = anchors[j].index
        end = anchors[j + 1].index if j + 1 < len(anchors) else T
        trajectory[start:end] = anchors[j].s

    return trajectory


def run_experiment(
    cfg: ExperimentConfig | None = None,
    verbose: bool = False,
) -> ExperimentReport:
    if cfg is None:
        cfg = load_config("experiments/configs/default.yaml")
        validate(cfg)

    ov = get_experiment_overrides(cfg, "exp04_exact_reconstruction")
    dims       = ov.get("dims", [1, 3, 10])
    T_values   = ov.get("T_values", [50, 200, 500])
    num_trials = ov.get("num_trials", 50)
    recon_atol = cfg.verification.reconstruction_atol

    K = cfg.memory_operator.K
    r = cfg.memory_operator.r
    tau = cfg.memory_operator.tau

    capsule = create_run_capsule(cfg.output_dir, "exact_reconstruction")
    save_config_snapshot(capsule, cfg)
    apply_theme(cfg.plotting.theme)
    setup_run_logging(capsule, verbose=verbose, experiment_id=EXPERIMENT_ID)

    log.info("[%s] %s", EXPERIMENT_ID, EXPERIMENT_NAME)
    log.info("  dims=%s  T=%s  trials=%d  atol=%.2e", dims, T_values, num_trials, recon_atol)

    rng = np.random.default_rng(cfg.execution.seed)

    report = ExperimentReport(
        experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
        formal_reference=FORMAL_REF, claim=CLAIM,
    )
    csv_rows: list[dict] = []
    sample_fig_done = False

    with ExperimentTimer() as timer:
        sweep = [(d, T, m)
                 for d in dims for T in T_values
                 for m in [1, 2, 5, min(10, K)]]

        for d, T, m_target in iter_progress(sweep, desc="reconstruction"):
            spacing = r + 3
            change_points = [1 + i * spacing for i in range(m_target)]
            change_points = [cp for cp in change_points if cp < T]
            if not change_points:
                continue

            all_exact = True
            max_error = 0.0
            error_msg = None
            valid_trials = 0
            skipped_trials = 0
            attempts = 0

            while valid_trials < num_trials:
                attempts += 1
                if attempts > max(10 * num_trials, 1000):
                    all_exact = False
                    error_msg = (
                        f"Could only realize {valid_trials}/{num_trials} valid trials "
                        f"under theorem assumptions"
                    )
                    break

                seed_t = rng.integers(0, 2**31)
                traj, _ = piecewise_constant(d, T, change_points,
                                             jump_magnitude=tau * 20, seed=seed_t)
                scores = sharp_event_score(traj)

                cp_ok = all(scores[cp] >= tau for cp in change_points)
                non_cp_ok = all(
                    scores[t] < tau for t in range(1, T) if t not in change_points
                )
                if not (cp_ok and non_cp_ok):
                    skipped_trials += 1
                    continue

                indices, anchors = select_anchors(scores, traj, K, r, tau)
                if indices != change_points:
                    skipped_trials += 1
                    continue  # Thm 8.3 failed -- tested in EXP-03

                reconstructed = reconstruct_trajectory(traj[0], anchors, T)
                error = float(np.max(np.abs(reconstructed - traj)))
                max_error = max(max_error, error)

                if error > recon_atol:
                    all_exact = False
                    error_msg = f"Validated trial {valid_trials}: max error = {error:.2e}"
                    break
                valid_trials += 1

                # Sample figure
                if not sample_fig_done and d == dims[0] and m_target >= 2:
                    plot_reconstruction_error(
                        traj, reconstructed, indices,
                        title=f"{EXPERIMENT_ID}: d={d}, m={m_target}",
                        save_path=capsule.figures / "reconstruction_sample",
                        formats=cfg.plotting.formats, theme=cfg.plotting.theme,
                    )
                    sample_fig_done = True
                    log.info("Figure saved -> %s/reconstruction_sample", capsule.figures)

            case_name = f"d{d}_T{T}_m{m_target}"
            report.add_case(TestCase(
                name=case_name, passed=all_exact,
                details={"d": d, "T": T, "m_target": m_target,
                         "max_error": max_error, "num_trials": num_trials,
                         "valid_trials": valid_trials, "skipped_trials": skipped_trials},
                error=error_msg,
            ))

            row = {"d": d, "T": T, "m_target": m_target,
                   "num_trials": num_trials, "valid_trials": valid_trials,
                   "skipped_trials": skipped_trials, "max_error": max_error,
                   "passed": all_exact}
            csv_rows.append(row)
            append_metrics_jsonl(
                {"case": case_name, **row}, capsule.metrics / "metrics.jsonl",
            )

            if verbose:
                icon = "PASS" if all_exact else "FAIL"
                print(f"  [{icon}] {case_name}: err={max_error:.2e}")

        # Edge: constant trajectory
        for d in [1, 5]:
            traj = np.tile(rng.standard_normal(d), (100, 1))
            scores = sharp_event_score(traj)
            indices, anchors = select_anchors(scores, traj, K, r, tau)
            reconstructed = reconstruct_trajectory(traj[0], anchors, 100)
            error = float(np.max(np.abs(reconstructed - traj)))
            report.add_case(TestCase(
                name=f"constant_d{d}", passed=error < recon_atol,
                details={"error": error},
                error=None if error < recon_atol else f"Error={error:.2e}",
            ))
            csv_rows.append({
                "d": d, "T": 100, "m_target": 0, "num_trials": 1,
                "valid_trials": 1, "skipped_trials": 0, "max_error": error,
                "passed": error < recon_atol,
            })
            append_metrics_jsonl(
                {
                    "case": f"constant_d{d}",
                    "d": d, "T": 100, "m_target": 0, "num_trials": 1,
                    "valid_trials": 1, "skipped_trials": 0, "max_error": error,
                    "passed": error < recon_atol,
                },
                capsule.metrics / "metrics.jsonl",
            )

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
