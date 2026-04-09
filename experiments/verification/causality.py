"""
Experiment 01 -- Causality Verification
========================================

Formal claim
    Theorem 8.1  (docs/formal_math/02_rigorous_architecture.md)
    If the maps defining the event score are causal, then M is causal.
    M(x_{1:t}) depends only on x_{1:t}.

Methodology
    For a trajectory x_{1:T}, pick a probe time t < T.
    Create N perturbed copies where x_{t+1:T} is replaced with fresh noise.
    Verify:
      - Event scores e_1, ..., e_t are bit-identical across all perturbations.
      - Hysteretic latent states h_1, ..., h_t are identical.

Outputs
    figures/
        causality_event_scores.{png,pdf}
    metrics/
        metrics.jsonl         -- per-case PASS/FAIL
        results.csv           -- flat table for publication
    artifacts/
        report.json           -- structured report
    logs/
        run.log               -- full session log
"""

from __future__ import annotations

import logging
import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synapse_core.event_encoder import sharp_event_score, hysteretic_event_score
from synapse_core.anchor_selector import select_anchors
from experiments.utils.config import load_config, validate, ExperimentConfig, get_experiment_overrides
from experiments.utils.model_io import (
    create_run_capsule, save_config_snapshot, save_metrics, save_run_pointer,
)
from experiments.utils.logging import setup_run_logging
from experiments.utils.exporter import append_metrics_jsonl, save_results_csv
from experiments.utils.progress import iter_progress
from experiments.common.report import (
    ExperimentReport, TestCase, ExperimentTimer, print_report, save_report_json,
)
from experiments.common.theme import apply_theme
from experiments.common.plotting import plot_event_scores

log = logging.getLogger(__name__)

EXPERIMENT_ID = "EXP-01"
EXPERIMENT_NAME = "Causality Verification"
FORMAL_REF = "Theorem 8.1 (02_rigorous_architecture.md)"
CLAIM = "M(x_{1:t}) depends only on x_{1:t}"

# CSV column order for the publication table
CSV_FIELDS = [
    "encoder", "d", "T", "t_probe", "alpha",
    "num_perturbations", "passed", "max_diff",
]


def run_experiment(
    cfg: ExperimentConfig | None = None,
    verbose: bool = False,
) -> ExperimentReport:
    """Run causality verification.

    Parameters
    ----------
    cfg : ExperimentConfig, optional
        Loaded and validated config.  Falls back to default YAML.
    verbose : bool
        Print per-case results to console.

    Returns
    -------
    ExperimentReport
    """
    # ---- config resolution -------------------------------------------------
    if cfg is None:
        cfg = load_config("experiments/configs/default.yaml")
        validate(cfg)

    ov = get_experiment_overrides(cfg, "exp01_causality")

    dims              = ov.get("dims", [2, 5, 10])
    lengths           = ov.get("lengths", [50, 100, 200])
    num_perturbations = ov.get("num_perturbations", 50)
    alpha_values      = ov.get("alpha_values", [0.3, 0.7])
    score_atol        = cfg.verification.score_match_atol

    K   = cfg.memory_operator.K
    r   = cfg.memory_operator.r
    tau = cfg.memory_operator.tau
    seed = cfg.execution.seed

    # ---- capsule + logging -------------------------------------------------
    capsule = create_run_capsule(cfg.output_dir, "causality")
    save_config_snapshot(capsule, cfg)
    apply_theme(cfg.plotting.theme)
    setup_run_logging(capsule, verbose=verbose, experiment_id=EXPERIMENT_ID)

    log.info("[%s] %s", EXPERIMENT_ID, EXPERIMENT_NAME)
    log.info("  dims=%s  lengths=%s  perturbations=%d", dims, lengths, num_perturbations)
    log.info("  K=%d  r=%d  tau=%.4f  seed=%d", K, r, tau, seed)

    rng = np.random.default_rng(seed)

    report = ExperimentReport(
        experiment_id=EXPERIMENT_ID,
        experiment_name=EXPERIMENT_NAME,
        formal_reference=FORMAL_REF,
        claim=CLAIM,
    )
    csv_rows: list[dict] = []

    # ---- Test A: Sharp encoder causality -----------------------------------
    with ExperimentTimer() as timer:
        sharp_configs = [(d, T, tp)
                         for d in dims for T in lengths
                         for tp in (T // 4, T // 2, 3 * T // 4)
                         if 1 < tp < T]

        for d, T, t_probe in iter_progress(sharp_configs, desc="sharp causality"):
            base_traj = rng.standard_normal((T, d))
            truncated = base_traj[:t_probe + 1].copy()
            ref_scores = sharp_event_score(truncated)

            all_match = True
            max_diff = 0.0
            error_msg = None

            for p in range(num_perturbations):
                perturbed = base_traj.copy()
                perturbed[t_probe + 1:] = rng.standard_normal((T - t_probe - 1, d))
                full_scores = sharp_event_score(perturbed)

                diff = np.max(np.abs(ref_scores - full_scores[:t_probe + 1]))
                max_diff = max(max_diff, diff)

                if diff > score_atol:
                    all_match = False
                    error_msg = f"Perturbation {p}: max_diff={diff:.2e}"
                    break

            case_name = f"sharp_d{d}_T{T}_t{t_probe}"
            report.add_case(TestCase(
                name=case_name,
                passed=all_match,
                details={"d": d, "T": T, "t_probe": t_probe,
                         "num_perturbations": num_perturbations,
                         "max_diff": float(max_diff)},
                error=error_msg,
            ))

            row = {"encoder": "sharp", "d": d, "T": T, "t_probe": t_probe,
                   "alpha": None, "num_perturbations": num_perturbations,
                   "passed": all_match, "max_diff": float(max_diff)}
            csv_rows.append(row)

            append_metrics_jsonl(
                {"test": "sharp", "case": case_name, **row},
                capsule.metrics / "metrics.jsonl",
            )

            if verbose:
                icon = "PASS" if all_match else "FAIL"
                print(f"  [{icon}] {case_name}  max_diff={max_diff:.2e}")

        # ---- Test B: Hysteretic encoder causality --------------------------
        hyst_configs = [(d, T, alpha)
                        for d in dims for T in [50, 100]
                        for alpha in alpha_values]

        for d, T, alpha in iter_progress(hyst_configs, desc="hysteretic causality"):
            t_probe = T // 2
            base_traj = rng.standard_normal((T, d))
            truncated = base_traj[:t_probe + 1].copy()
            ref_scores, ref_h = hysteretic_event_score(truncated, alpha=alpha)

            all_match = True
            max_diff = 0.0
            error_msg = None

            for p in range(num_perturbations):
                perturbed = base_traj.copy()
                perturbed[t_probe + 1:] = rng.standard_normal((T - t_probe - 1, d))
                full_scores, full_h = hysteretic_event_score(perturbed, alpha=alpha)

                diff_s = np.max(np.abs(ref_scores - full_scores[:t_probe + 1]))
                diff_h = np.max(np.abs(ref_h - full_h[:t_probe + 1]))
                diff = max(diff_s, diff_h)
                max_diff = max(max_diff, diff)

                if diff_s > score_atol:
                    all_match = False
                    error_msg = f"Hysteretic score mismatch: max_diff={diff_s:.2e}"
                    break
                if diff_h > score_atol:
                    all_match = False
                    error_msg = f"Hysteretic latent state mismatch: max_diff={diff_h:.2e}"
                    break

            case_name = f"hysteretic_d{d}_T{T}_alpha{alpha}"
            report.add_case(TestCase(
                name=case_name,
                passed=all_match,
                details={"d": d, "T": T, "alpha": alpha,
                         "num_perturbations": num_perturbations,
                         "max_diff": float(max_diff)},
                error=error_msg,
            ))

            row = {"encoder": "hysteretic", "d": d, "T": T, "t_probe": t_probe,
                   "alpha": alpha, "num_perturbations": num_perturbations,
                   "passed": all_match, "max_diff": float(max_diff)}
            csv_rows.append(row)

            append_metrics_jsonl(
                {"test": "hysteretic", "case": case_name, **row},
                capsule.metrics / "metrics.jsonl",
            )

            if verbose:
                icon = "PASS" if all_match else "FAIL"
                print(f"  [{icon}] {case_name}  max_diff={max_diff:.2e}")

    # ---- finalise ----------------------------------------------------------
    report.duration_seconds = timer.elapsed
    report.finalize()

    # -- save structured outputs
    save_report_json(report, str(capsule.artifacts / "report.json"))
    save_results_csv(csv_rows, CSV_FIELDS, capsule.metrics / "results.csv")

    # -- generate publication figure
    sample_traj = rng.standard_normal((50, 3))
    sample_scores = sharp_event_score(sample_traj)
    sample_idx, _ = select_anchors(sample_scores, sample_traj, K, r, tau)
    plot_event_scores(
        sample_traj, sample_scores,
        anchor_indices=sample_idx, tau=tau,
        title=f"{EXPERIMENT_ID}: Event Score Causality",
        save_path=capsule.figures / "causality_event_scores",
        formats=cfg.plotting.formats,
        theme=cfg.plotting.theme,
    )
    log.info("Figure saved -> %s/causality_event_scores", capsule.figures)

    save_run_pointer(capsule, cfg.output_dir)
    log.info("[%s] %s  (%d/%d)", EXPERIMENT_ID, report.status,
             report.passed_cases, report.total_cases)

    return report


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
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
