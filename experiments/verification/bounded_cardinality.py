"""
Experiment 02 -- Bounded Anchor Cardinality
============================================

Formal claim
    Theorem 8.2  (docs/formal_math/02_rigorous_architecture.md)
    For every x_{1:T}, |I*(x_{1:T})| <= K.

Methodology
    Sweep K, r, tau, T.  For each config run adversarial, random-walk,
    and piecewise-constant trajectories.  Verify |I*| <= K unconditionally.

Outputs
    figures/cardinality_vs_K.{png,pdf}
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
from experiments.common.trajectory_generators import (
    adversarial_dense_events, random_walk, piecewise_constant,
)
from experiments.common.report import (
    ExperimentReport, TestCase, ExperimentTimer, print_report, save_report_json,
)
from experiments.common.theme import apply_theme
from experiments.common.plotting import plot_cardinality_vs_K

log = logging.getLogger(__name__)

EXPERIMENT_ID = "EXP-02"
EXPERIMENT_NAME = "Bounded Anchor Cardinality"
FORMAL_REF = "Theorem 8.2 (02_rigorous_architecture.md)"
CLAIM = "|I*(x_{1:T})| <= K for all trajectories"

CSV_FIELDS = ["K", "r", "tau", "T", "num_trials", "max_m_observed", "passed"]


def run_experiment(
    cfg: ExperimentConfig | None = None,
    verbose: bool = False,
) -> ExperimentReport:
    if cfg is None:
        cfg = load_config("experiments/configs/default.yaml")
        validate(cfg)

    ov = get_experiment_overrides(cfg, "exp02_bounded_cardinality")
    K_values   = ov.get("K_values", [1, 5, 10, 50])
    r_values   = ov.get("r_values", [0, 1, 5, 10])
    tau_values = ov.get("tau_values", [0.01, 0.1, 1.0])
    T_values   = ov.get("T_values", [100, 500])
    num_trials = ov.get("num_trials", 100)
    d = cfg.trajectory.d

    capsule = create_run_capsule(cfg.output_dir, "bounded_cardinality")
    save_config_snapshot(capsule, cfg)
    apply_theme(cfg.plotting.theme)
    setup_run_logging(capsule, verbose=verbose, experiment_id=EXPERIMENT_ID)

    log.info("[%s] %s", EXPERIMENT_ID, EXPERIMENT_NAME)
    log.info("  K=%s  r=%s  tau=%s  T=%s  trials=%d  d=%d",
             K_values, r_values, tau_values, T_values, num_trials, d)

    rng = np.random.default_rng(cfg.execution.seed)

    report = ExperimentReport(
        experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
        formal_reference=FORMAL_REF, claim=CLAIM,
    )
    csv_rows: list[dict] = []
    max_m_per_K: dict[int, int] = {}

    with ExperimentTimer() as timer:
        configs = [(K, r, tau, T)
                   for K in K_values for r in r_values
                   for tau in tau_values for T in T_values]

        for K, r, tau, T in iter_progress(configs, desc="cardinality sweep"):
            max_m_per_K.setdefault(K, 0)
            all_bounded = True
            max_m = 0
            error_msg = None

            for trial in range(num_trials):
                seed_t = rng.integers(0, 2**31)

                if trial % 3 == 0:
                    traj = adversarial_dense_events(
                        d, T, K, r, tau, event_magnitude=tau * 10, seed=seed_t)
                elif trial % 3 == 1:
                    traj = random_walk(d, T, step_std=tau * 5, seed=seed_t)
                else:
                    n_cp = max(1, min(K * 3, (T - 2) // max(r + 1, 1)))
                    cps = sorted(rng.choice(
                        range(1, T - 1), size=min(n_cp, T - 2), replace=False,
                    ).tolist())
                    traj, _ = piecewise_constant(d, T, cps,
                                                 jump_magnitude=tau * 10,
                                                 seed=seed_t)

                scores = sharp_event_score(traj)
                indices, _ = select_anchors(scores, traj, K, r, tau)
                m = len(indices)
                max_m = max(max_m, m)

                if m > K:
                    all_bounded = False
                    error_msg = f"Trial {trial}: |I*|={m} > K={K}"
                    break

            max_m_per_K[K] = max(max_m_per_K[K], max_m)

            case_name = f"K{K}_r{r}_tau{tau}_T{T}"
            report.add_case(TestCase(
                name=case_name, passed=all_bounded,
                details={"K": K, "r": r, "tau": tau, "T": T,
                         "max_m_observed": max_m, "num_trials": num_trials},
                error=error_msg,
            ))

            row = {"K": K, "r": r, "tau": tau, "T": T,
                   "num_trials": num_trials, "max_m_observed": max_m,
                   "passed": all_bounded}
            csv_rows.append(row)
            append_metrics_jsonl(
                {"case": case_name, **row},
                capsule.metrics / "metrics.jsonl",
            )

            if verbose:
                icon = "PASS" if all_bounded else "FAIL"
                print(f"  [{icon}] {case_name}: max_m={max_m} <= K={K}")

        # -- Saturation check -----------------------------------------------
        for K, r, tau, T in [(10, 2, 0.1, 200), (5, 1, 0.5, 100)]:
            cps = [1 + i * (r + 2) for i in range(K)]
            cps = [cp for cp in cps if cp < T]
            if len(cps) < K:
                continue

            traj, _ = piecewise_constant(d, T, cps,
                                         jump_magnitude=tau * 20, seed=123)
            scores = sharp_event_score(traj)
            indices, _ = select_anchors(scores, traj, K, r, tau)
            passed = len(indices) == len(cps) and len(indices) <= K

            report.add_case(TestCase(
                name=f"saturation_K{K}_r{r}",
                passed=passed,
                details={"expected": len(cps), "actual": len(indices), "K": K},
                error=None if passed else f"|I*|={len(indices)} != {len(cps)}",
            ))

    report.duration_seconds = timer.elapsed
    report.finalize()

    save_report_json(report, str(capsule.artifacts / "report.json"))
    save_results_csv(csv_rows, CSV_FIELDS, capsule.metrics / "results.csv")

    Ks = sorted(max_m_per_K.keys())
    plot_cardinality_vs_K(
        Ks, [max_m_per_K[k] for k in Ks],
        save_path=capsule.figures / "cardinality_vs_K",
        formats=cfg.plotting.formats, theme=cfg.plotting.theme,
    )
    log.info("Figure saved -> %s/cardinality_vs_K", capsule.figures)

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
