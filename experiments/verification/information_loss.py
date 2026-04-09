"""
Experiment 07 -- Information Loss
===================================

Formal claim
    Proposition 8.6  (docs/formal_math/02_rigorous_architecture.md)
    There exist distinct x != y with M(x) = M(y).
    Collision rate decreases as K increases.

Outputs
    figures/collision_rates.{png,pdf}
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
from synapse_core.memory_operator import M
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
from experiments.common.plotting import plot_collision_rates

log = logging.getLogger(__name__)

EXPERIMENT_ID = "EXP-07"
EXPERIMENT_NAME = "Information Loss"
FORMAL_REF = "Proposition 8.6 (02_rigorous_architecture.md)"
CLAIM = "There exist distinct x != y with M(x) = M(y)"

CSV_FIELDS = ["K", "num_trajectories", "collisions", "collision_rate"]


def _memory_fingerprint(result: dict) -> tuple:
    """Create a hashable fingerprint of a memory operator result."""
    anchors = result["anchor_indices"]
    if len(anchors) > 0:
        scores = tuple(round(float(s), 12)
                       for s in result["event_scores"][anchors])
    else:
        scores = ()
    return (tuple(anchors), scores)


def run_experiment(
    cfg: ExperimentConfig | None = None,
    verbose: bool = False,
) -> ExperimentReport:
    if cfg is None:
        cfg = load_config("experiments/configs/default.yaml")
        validate(cfg)

    ov = get_experiment_overrides(cfg, "exp07_information_loss")
    num_collision_attempts = ov.get("num_collision_attempts", 100)
    num_trajectories       = ov.get("num_trajectories", 200)
    K_sweep                = ov.get("K_sweep", [1, 3, 5, 10, 20])

    d       = cfg.trajectory.d
    r       = cfg.memory_operator.r
    tau     = cfg.memory_operator.tau
    weights = cfg.memory_operator.weights.as_tuple()
    Q       = cfg.memory_operator.Q

    capsule = create_run_capsule(cfg.output_dir, "information_loss")
    save_config_snapshot(capsule, cfg)
    apply_theme(cfg.plotting.theme)
    setup_run_logging(capsule, verbose=verbose, experiment_id=EXPERIMENT_ID)

    log.info("[%s] %s", EXPERIMENT_ID, EXPERIMENT_NAME)
    log.info("  d=%d  r=%d  tau=%.4f  Q=%d", d, r, tau, Q)
    log.info("  collision_attempts=%d  trajectories=%d  K_sweep=%s",
             num_collision_attempts, num_trajectories, K_sweep)

    rng = np.random.default_rng(cfg.execution.seed)

    report = ExperimentReport(
        experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
        formal_reference=FORMAL_REF, claim=CLAIM,
    )
    csv_rows: list[dict] = []

    with ExperimentTimer() as timer:
        # ---- Test A: Constructive collision --------------------------------
        collision_found = False

        for trial in iter_progress(range(num_collision_attempts),
                                   desc="constructive collision"):
            K_c, T_c = 3, 50
            cps = [10, 25, 40]
            traj_x, _ = piecewise_constant(
                d, T_c, cps, jump_magnitude=tau * 20,
                seed=rng.integers(0, 2**31),
            )
            traj_y = traj_x.copy()
            # Perturb non-anchor positions by tiny amount
            for i in range(T_c):
                if i not in cps:
                    traj_y[i] += rng.standard_normal(d) * tau * 0.01

            scores_x = sharp_event_score(traj_x)
            scores_y = sharp_event_score(traj_y)
            idx_x, _ = select_anchors(scores_x, traj_x, K_c, r, tau)
            idx_y, _ = select_anchors(scores_y, traj_y, K_c, r, tau)

            if idx_x == idx_y:
                res_x = M(traj_x, K=K_c, r=r, tau=tau, weights=weights, Q=Q)
                res_y = M(traj_y, K=K_c, r=r, tau=tau, weights=weights, Q=Q)
                fp_x = _memory_fingerprint(res_x)
                fp_y = _memory_fingerprint(res_y)
                if fp_x == fp_y and not np.array_equal(traj_x, traj_y):
                    collision_found = True
                    log.info("  Collision found at trial %d", trial)
                    break

        report.add_case(TestCase(
            name="constructive_collision", passed=collision_found,
            details={"attempts": num_collision_attempts},
            error=None if collision_found else "No collision found",
        ))
        append_metrics_jsonl(
            {"test": "constructive", "passed": collision_found},
            capsule.metrics / "metrics.jsonl",
        )
        if verbose:
            print(f"  [{'PASS' if collision_found else 'FAIL'}] constructive_collision")

        # ---- Test B: Collision rate vs K -----------------------------------
        collision_rates: list[float] = []

        for K in iter_progress(K_sweep, desc="collision sweep"):
            T_s = 50
            collisions = 0
            fingerprints: dict[tuple, int] = {}

            for i in range(num_trajectories):
                traj = rng.standard_normal((T_s, d))
                res = M(traj, K=K, r=r, tau=tau, weights=weights, Q=Q)
                fp = _memory_fingerprint(res)
                if fp in fingerprints:
                    collisions += 1
                fingerprints[fp] = i

            rate = collisions / max(num_trajectories, 1)
            collision_rates.append(rate)

            report.add_case(TestCase(
                name=f"collision_rate_K{K}", passed=True,
                details={"K": K, "collisions": collisions, "rate": round(rate, 6)},
            ))

            row = {"K": K, "num_trajectories": num_trajectories,
                   "collisions": collisions, "collision_rate": round(rate, 6)}
            csv_rows.append(row)
            append_metrics_jsonl(
                {"test": f"rate_K{K}", **row},
                capsule.metrics / "metrics.jsonl",
            )
            if verbose:
                print(f"  [PASS] collision_rate_K{K}: {rate:.4f}")

        # ---- Test C: Monotonicity ------------------------------------------
        if len(collision_rates) >= 2:
            is_monotone = all(
                collision_rates[i] >= collision_rates[i + 1] - 0.05
                for i in range(len(collision_rates) - 1)
            )
            report.add_case(TestCase(
                name="monotonicity_K_vs_collision", passed=is_monotone,
                details={"rates": [round(r, 6) for r in collision_rates],
                         "K_values": K_sweep},
                error=None if is_monotone else f"Non-monotone: {collision_rates}",
            ))

    report.duration_seconds = timer.elapsed
    report.finalize()

    save_report_json(report, str(capsule.artifacts / "report.json"))
    save_results_csv(csv_rows, CSV_FIELDS, capsule.metrics / "results.csv")

    plot_collision_rates(
        K_sweep, collision_rates,
        save_path=capsule.figures / "collision_rates",
        formats=cfg.plotting.formats, theme=cfg.plotting.theme,
    )
    log.info("Figure saved -> %s/collision_rates", capsule.figures)

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
