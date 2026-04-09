"""
Experiment 08 -- End-to-End Integration
==========================================

Formal claim
    Section 8 of 01_main_definition.md
    M(x_{1:T}) = (A, Dgm_0,...,Dgm_Q) is well-defined and computable
    for all valid inputs.

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

from synapse_core.memory_operator import M, MemoryState
from experiments.utils.config import load_config, validate, ExperimentConfig, get_experiment_overrides
from experiments.utils.model_io import create_run_capsule, save_config_snapshot, save_run_pointer
from experiments.utils.logging import setup_run_logging
from experiments.utils.exporter import append_metrics_jsonl, save_results_csv
from experiments.utils.progress import iter_progress
from experiments.common.trajectory_generators import random_walk, piecewise_constant
from experiments.common.report import (
    ExperimentReport, TestCase, ExperimentTimer, print_report, save_report_json,
)
from experiments.common.theme import apply_theme

log = logging.getLogger(__name__)

EXPERIMENT_ID = "EXP-08"
EXPERIMENT_NAME = "End-to-End Integration"
FORMAL_REF = "Section 8 of 01_main_definition.md"
CLAIM = "M(x_{1:T}) well-defined and computable for all valid inputs"

CSV_FIELDS = ["case_name", "category", "d", "T", "passed", "error"]


def _validate_result(
    result: MemoryState,
    traj: np.ndarray,
    K: int,
    tau: float,
    Q: int,
) -> tuple[bool, str | None]:
    """Validate M output structure and formal invariants."""
    indices = result.anchor_indices
    scores = result.event_scores
    diagrams = result.persistence_diagrams

    # No NaN/Inf in scores
    if scores is not None and (np.any(np.isnan(scores)) or np.any(np.isinf(scores))):
        return False, "NaN/Inf in event scores"

    # |I*| <= K
    if len(indices) > K:
        return False, f"|I*|={len(indices)} > K={K}"

    # All anchor scores >= tau
    if scores is not None:
        for idx in indices:
            if scores[idx] < tau - 1e-14:
                return False, f"Anchor score {scores[idx]:.6e} < tau={tau} at index {idx}"

    # Indices are sorted and in range
    if indices != sorted(indices):
        return False, f"Indices not sorted: {indices}"
    T = traj.shape[0]
    if indices and (indices[0] < 0 or indices[-1] >= T):
        return False, f"Index out of range: {indices}"

    # Persistence diagram: birth <= death
    if diagrams is not None:
        for q, dgm in enumerate(diagrams):
            for b, d_val in dgm:
                if np.isfinite(d_val) and b > d_val + 1e-14:
                    return False, f"Dgm_{q}: birth={b:.6e} > death={d_val:.6e}"

    return True, None


def run_experiment(
    cfg: ExperimentConfig | None = None,
    verbose: bool = False,
) -> ExperimentReport:
    if cfg is None:
        cfg = load_config("experiments/configs/default.yaml")
        validate(cfg)

    ov = get_experiment_overrides(cfg, "exp08_end_to_end_integration")
    dims_normal  = ov.get("dims_normal", [1, 3, 10])
    dims_edge    = ov.get("dims_edge", [1, 5, 50])
    alpha_values = ov.get("alpha_values", [0.3, 0.7])

    K       = cfg.memory_operator.K
    r       = cfg.memory_operator.r
    tau     = cfg.memory_operator.tau
    weights = cfg.memory_operator.weights.as_tuple()
    Q       = cfg.memory_operator.Q

    capsule = create_run_capsule(cfg.output_dir, "end_to_end")
    save_config_snapshot(capsule, cfg)
    apply_theme(cfg.plotting.theme)
    setup_run_logging(capsule, verbose=verbose, experiment_id=EXPERIMENT_ID)

    log.info("[%s] %s", EXPERIMENT_ID, EXPERIMENT_NAME)
    log.info("  K=%d  r=%d  tau=%.4f  Q=%d", K, r, tau, Q)
    log.info("  dims_normal=%s  dims_edge=%s  alpha=%s",
             dims_normal, dims_edge, alpha_values)

    rng = np.random.default_rng(cfg.execution.seed)

    report = ExperimentReport(
        experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
        formal_reference=FORMAL_REF, claim=CLAIM,
    )
    csv_rows: list[dict] = []

    with ExperimentTimer() as timer:
        # ---- Normal trajectories -------------------------------------------
        normal_sweep = [(d, T, tt)
                        for d in dims_normal for T in [50, 100, 200]
                        for tt in ["random_walk", "piecewise_constant"]]

        for d, T, traj_type in iter_progress(normal_sweep, desc="normal"):
            if traj_type == "random_walk":
                traj = random_walk(d, T, step_std=1.0,
                                   seed=rng.integers(0, 2**31))
            else:
                n_cp = min(5, T // 10)
                cps = sorted(rng.choice(
                    range(2, T - 1), size=n_cp, replace=False,
                ).tolist())
                traj, _ = piecewise_constant(
                    d, T, cps, jump_magnitude=tau * 10,
                    seed=rng.integers(0, 2**31),
                )

            result = M(traj, K=K, r=r, tau=tau, weights=weights, Q=Q)
            valid, err = _validate_result(result, traj, K, tau, Q)

            case_name = f"normal_{traj_type}_d{d}_T{T}"
            report.add_case(TestCase(
                name=case_name, passed=valid,
                details={"d": d, "T": T, "type": traj_type},
                error=err,
            ))

            row = {"case_name": case_name, "category": "normal",
                   "d": d, "T": T, "passed": valid, "error": err or ""}
            csv_rows.append(row)
            append_metrics_jsonl(
                {"case": case_name, **row}, capsule.metrics / "metrics.jsonl",
            )
            if verbose:
                print(f"  [{'PASS' if valid else 'FAIL'}] {case_name}")

        # ---- Edge cases ----------------------------------------------------
        edge_cases: list[tuple[str, np.ndarray | None, dict]] = [
            ("T1_scalar", np.array([[1.0]]),
             {"K": K, "r": r, "tau": tau}),
            ("T2_minimal", rng.standard_normal((2, 3)),
             {"K": K, "r": r, "tau": tau}),
            ("constant", np.tile([1.0, 2.0, 3.0], (100, 1)),
             {"K": K, "r": r, "tau": tau}),
            ("K0", rng.standard_normal((50, 3)),
             {"K": 0, "r": r, "tau": tau}),
        ]
        for d_e in dims_edge:
            edge_cases.append(
                (f"high_dim_d{d_e}", rng.standard_normal((50, d_e)),
                 {"K": K, "r": r, "tau": tau})
            )

        for case_label, traj, params in iter_progress(edge_cases, desc="edge cases"):
            K_e = params["K"]
            result = M(traj, K=K_e, r=params["r"], tau=params["tau"],
                       weights=weights, Q=Q)

            if case_label == "K0":
                valid = len(result.anchor_indices) == 0
                err = None if valid else "K=0 but anchors selected"
            else:
                valid, err = _validate_result(result, traj, K_e, params["tau"], Q)

            full_name = f"edge_{case_label}"
            report.add_case(TestCase(
                name=full_name, passed=valid,
                details={"case": case_label}, error=err,
            ))

            row = {"case_name": full_name, "category": "edge",
                   "d": traj.shape[1], "T": traj.shape[0],
                   "passed": valid, "error": err or ""}
            csv_rows.append(row)
            append_metrics_jsonl(
                {"case": full_name, **row}, capsule.metrics / "metrics.jsonl",
            )
            if verbose:
                print(f"  [{'PASS' if valid else 'FAIL'}] {full_name}")

        # ---- Hysteretic mode -----------------------------------------------
        for alpha in alpha_values:
            traj = rng.standard_normal((100, 5))
            result = M(traj, K=K, r=r, tau=tau, weights=weights, Q=Q, alpha=alpha)
            valid, err = _validate_result(result, traj, K, tau, Q)

            case_name = f"hysteretic_alpha{alpha}"
            report.add_case(TestCase(
                name=case_name, passed=valid,
                details={"alpha": alpha}, error=err,
            ))

            row = {"case_name": case_name, "category": "hysteretic",
                   "d": 5, "T": 100, "passed": valid, "error": err or ""}
            csv_rows.append(row)
            append_metrics_jsonl(
                {"case": case_name, **row}, capsule.metrics / "metrics.jsonl",
            )
            if verbose:
                print(f"  [{'PASS' if valid else 'FAIL'}] {case_name}")

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
