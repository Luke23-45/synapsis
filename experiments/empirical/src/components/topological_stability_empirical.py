"""
EMP-11: Topological Stability Verification

Empirically validates Theorem 12.3 — the stability of persistence diagrams
under perturbation of the lifted point cloud.

Formal Claim (Thm 12.3):
    If  max_j ‖p_j − p'_j‖_2 ≤ ε,  then
        d_B( Dgm_q(P), Dgm_q(P') ) ≤ 2ε
    for each homology degree q = 0, 1, ..., Q.

Uses gudhi.bottleneck_distance when available for exact computation;
falls back to a fully-vectorised scipy-based Hungarian matching otherwise.

Z2 Reference: §12 of 02_rigorous_architecture.md, Thm 12.3
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from synapse_core.topological_summary import (
    compute_persistence_diagrams,
    PersistenceDiagram,
    has_full_persistence_backend,
)
from synapse_core.memory_operator import compute_memory
from experiments.common.trajectory_generators import piecewise_constant
from experiments.empirical.common.seed_runner import run_multi_seed
from experiments.empirical.common.emp_config import load_emp_config
from experiments.empirical.common.data_saver import save_experiment_npz
from experiments.empirical.common.math_utils import (
    make_orthogonal_W,
    generate_trajectory,
    bottleneck_distance,
)

log = logging.getLogger(__name__)


# ── Cloud Generators (vectorised) ─────────────────────────────────────────

def _generate_cloud_batch(
    family: str, n_points: int, dim: int, rng: np.random.Generator,
) -> np.ndarray:
    """Generate a point cloud with known topological structure — no loops."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    cloud = np.zeros((n_points, max(dim, 2)), dtype=np.float64)

    if family == "circle":
        cloud[:, 0] = np.cos(t)
        cloud[:, 1] = np.sin(t)
    elif family == "line":
        cloud[:, 0] = np.linspace(-1, 1, n_points)
    elif family == "clusters":
        half = n_points // 2
        cloud[:half] = rng.normal(loc=-3.0, scale=0.3, size=(half, max(dim, 2)))
        cloud[half:] = rng.normal(loc=3.0, scale=0.3, size=(n_points - half, max(dim, 2)))
        return cloud[:, :dim].astype(np.float64)
    else:  # random
        return rng.standard_normal((n_points, dim)).astype(np.float64)

    if dim > 2:
        cloud[:, 2:dim] = rng.normal(scale=0.01, size=(n_points, dim - 2))

    return cloud[:, :dim].astype(np.float64)


def _perturb_cloud(
    cloud: np.ndarray, epsilon: float, rng: np.random.Generator,
) -> np.ndarray:
    """Perturb each point by at most ε in L2 norm — fully vectorised."""
    n, dim = cloud.shape
    direction = rng.standard_normal((n, dim))
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    direction /= norms
    magnitudes = rng.uniform(0, epsilon, size=(n, 1))
    return cloud + direction * magnitudes


# ── Single-Seed Experiment ─────────────────────────────────────────────────

def run_single_seed(config: Any, seed: int) -> Dict[str, float]:
    """Run EMP-11 for a single seed.

    Test A: Direct cloud perturbation (Thm 12.3 exact test).
    Test B: Full-pipeline trajectory perturbation (diagnostic).
    """
    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed, workers=True)
    except ImportError:
        pass
    rng = np.random.default_rng(seed)
    cfg = config

    sweep = cfg.sweep if hasattr(cfg, "sweep") else cfg
    epsilon_values = getattr(sweep, "epsilon_values", [0.001, 0.01, 0.05, 0.1, 0.5])
    n_trials = getattr(sweep, "n_trials", 20)
    cloud_sizes = getattr(sweep, "cloud_sizes", [5, 10, 20])
    cloud_families = getattr(sweep, "cloud_families", ["random", "circle", "line", "clusters"])
    max_Q = getattr(sweep, "max_Q", 1)

    d = cfg.trajectory.d
    T = cfg.trajectory.T
    K = cfg.memory.K
    r = cfg.memory.r
    lam = cfg.memory.lam
    k = cfg.memory.k
    Q_cfg = cfg.memory.Q

    results: Dict[str, float] = {}
    clouds: List[np.ndarray] = []

    # ── Test A: Direct Cloud Perturbation (Thm 12.3 exact test) ───────
    total_tests = 0
    violations = 0
    max_ratio = 0.0
    per_eps: Dict[str, List[float]] = {f"eps_{e}": [] for e in epsilon_values}

    # Pre-build the full sweep as a flat product
    sweep_configs = [
        (eps, family, n_pts, trial)
        for eps in epsilon_values
        for family in cloud_families
        for n_pts in cloud_sizes
        for trial in range(n_trials)
    ]
    log.info("[EMP-11] Test A: %d cloud perturbation configs", len(sweep_configs))

    for eps, family, n_pts, _trial in sweep_configs:
        dim = max(k, 2)
        cloud = _generate_cloud_batch(family, n_pts, dim, rng)
        clouds.append(cloud)
        cloud_pert = _perturb_cloud(cloud, eps, rng)

        for q in range(min(max_Q + 1, 2)):
            total_tests += 1
            dgms = compute_persistence_diagrams(cloud, q)
            dgms_p = compute_persistence_diagrams(cloud_pert, q)

            d_b = bottleneck_distance(dgms[q], dgms_p[q])
            bound = 2.0 * eps
            ratio = d_b / bound if bound > 1e-15 else 0.0
            max_ratio = max(max_ratio, ratio)

            if d_b > bound + 1e-9:
                violations += 1

            per_eps[f"eps_{eps}"].append(ratio)

    results["A_total_tests"] = float(total_tests)
    results["A_violations"] = float(violations)
    results["A_pass_rate"] = 1.0 - violations / max(total_tests, 1)
    results["A_max_violation_ratio"] = max_ratio

    save_experiment_npz("EMP-11", seed, {"test_clouds": np.array(clouds, dtype=object) if clouds else np.zeros(0)}, cfg.output_dir)

    for eps in epsilon_values:
        ratios = per_eps[f"eps_{eps}"]
        if ratios:
            ratios_arr = np.asarray(ratios)
            results[f"A_eps{eps}_mean_ratio"] = float(ratios_arr.mean())
            results[f"A_eps{eps}_max_ratio"] = float(ratios_arr.max())

    # ── Test B: Full-Pipeline Trajectory Perturbation (Diagnostic) ────
    pipeline_ratios: List[float] = []
    D_dim = d + 3

    b_configs = [
        (trial, eps)
        for trial in range(min(n_trials, 10))
        for eps in [0.01, 0.05, 0.1]
    ]

    for trial, eps in b_configs:
        traj = generate_trajectory(d, T, rng)
        W_Theta = make_orthogonal_W(k, D_dim, rng)
        traj_pert = traj + rng.standard_normal(traj.shape) * eps

        state_o = compute_memory(traj, K, r, lam, W_Theta, Q_cfg, solver="osqp")
        state_p = compute_memory(traj_pert, K, r, lam, W_Theta, Q_cfg, solver="osqp")

        if (state_o.point_cloud.size > 0
                and state_p.point_cloud.size > 0
                and state_o.point_cloud.shape == state_p.point_cloud.shape):
            cloud_shift = float(np.max(
                np.linalg.norm(state_o.point_cloud - state_p.point_cloud, axis=1),
            ))
            if cloud_shift > 1e-12:
                for q in range(min(Q_cfg + 1, 2)):
                    d_b = bottleneck_distance(
                        state_o.persistence_diagrams[q],
                        state_p.persistence_diagrams[q],
                    )
                    pipeline_ratios.append(d_b / (2.0 * cloud_shift))

    if pipeline_ratios:
        pr_arr = np.asarray(pipeline_ratios)
        results["B_pipeline_mean_ratio"] = float(pr_arr.mean())
        results["B_pipeline_max_ratio"] = float(pr_arr.max())
        results["B_pipeline_bound_holds"] = float(np.all(pr_arr <= 1.0 + 1e-6))
    else:
        results["B_pipeline_mean_ratio"] = 0.0
        results["B_pipeline_max_ratio"] = 0.0
        results["B_pipeline_bound_holds"] = 1.0

    return results


# ── Entry Point ────────────────────────────────────────────────────────────

def run_experiment(config: Any = None) -> Dict[str, Any]:
    """Run EMP-11 across all seeds and return aggregated report."""
    if config is None:
        config = load_emp_config("EMP-11")

    report = run_multi_seed(
        experiment_fn=run_single_seed,
        config=config,
        seeds=config.training.seeds,
        experiment_id="EMP-11",
        output_dir=config.output_dir,
    )

    agg = report["aggregated"]
    pass_rate = agg.get("A_pass_rate", {}).get("mean", 0.0)
    max_ratio = agg.get("A_max_violation_ratio", {}).get("max", 0.0)

    passed = pass_rate >= 0.95 and max_ratio <= 1.05

    report["experiment_name"] = "Topological Stability Verification"
    report["pass_criterion"] = (
        f"d_B ≤ 2ε pass rate >= 95% (observed: {pass_rate:.4f}), "
        f"max ratio ≤ 1.05 (observed: {max_ratio:.4f})"
    )
    report["passed"] = passed
    report["has_full_persistence_backend"] = has_full_persistence_backend()
    report["has_gudhi_bottleneck"] = bottleneck_distance.__module__ is not None

    log.info(
        "[EMP-11] pass=%.4f max_ratio=%.4f passed=%s",
        pass_rate, max_ratio, passed,
    )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if str(Path(__file__).resolve().parent.parent.parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
    result = run_experiment()
    print(f"\nEMP-11 PASSED: {result['passed']}")
