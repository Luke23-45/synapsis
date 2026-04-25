"""
EMP-08: Hyperparameter Sensitivity Analysis

Systematically sweeps the three critical hyperparameters (λ, K, r) to quantify
how sensitive SYNAPSE performance is to their choice. Produces dense heatmaps
and degradation curves needed for publication robustness claims.

Z2 Reference: §5 (λ regularization), §6 (K budget, r refractory)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
try:
    import pytorch_lightning as pl
except ImportError:
    pl = None

from synapse_core.memory_operator import compute_memory
from synapse_core.event_encoder import sharp_event_score
from synapse_core.anchor_selector import solve_relaxed_selector, hard_projection
from experiments.common.trajectory_generators import piecewise_constant
from experiments.empirical.common.metrics import match_f1
from experiments.empirical.common.seed_runner import run_multi_seed
from experiments.empirical.common.emp_config import load_emp_config
from experiments.empirical.common.math_utils import make_orthogonal_W, ridge_probe_accuracy, pad_rows
from experiments.empirical.common.data_saver import save_experiment_npz

log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

# Helper imported from experiments.empirical.common.math_utils:
#   make_orthogonal_W


def _generate_labeled_trajectory(
    d: int, T: int, rng: np.random.Generator,
) -> Tuple[np.ndarray, List[int]]:
    """Generate trajectory with known sparse events for F1 evaluation."""
    n_events = int(rng.integers(3, 8))
    candidates = np.arange(10, T - 10)
    if len(candidates) < n_events:
        candidates = np.arange(2, T - 2)
    chosen = rng.choice(candidates, size=min(n_events, len(candidates)), replace=False)
    cps = sorted(chosen.tolist())
    traj, gt_cps = piecewise_constant(d, T, cps, jump_magnitude=5.0, seed=int(rng.integers(2**31)))
    traj = traj + rng.normal(scale=0.15, size=traj.shape)
    return traj.astype(np.float64), gt_cps


def _evaluate_recovery(
    traj: np.ndarray, gt_events: List[int],
    K: int, r: int, lam: float, tolerance: int = 2,
) -> Dict[str, float]:
    """Run the full Z2 selector pipeline and compute F1 + structural metrics."""
    scores = sharp_event_score(traj)
    y_star = solve_relaxed_selector(scores, K, r, lam, solver="osqp")
    detected = hard_projection(y_star, K, r)

    f1 = match_f1(detected, gt_events, tolerance)
    y_l2 = float(np.linalg.norm(y_star))
    y_sparsity = float(np.sum(y_star > 0.01)) / max(len(y_star), 1)
    n_detected = len(detected)

    return {
        "f1": f1,
        "n_detected": float(n_detected),
        "y_l2_norm": y_l2,
        "y_sparsity": y_sparsity,
    }


def _ridge_probe_from_memory(
    trajs_labels: List[Tuple[np.ndarray, int]],
    K: int, r: int, lam: float, k: int, Q: int,
    rng: np.random.Generator,
) -> float:
    """Quick ridge probe on memory state features for classification accuracy."""
    from experiments.empirical.common.baselines import summarize_diagrams

    D = trajs_labels[0][0].shape[1] + 3
    W_Theta = make_orthogonal_W(k, D, rng)

    features: List[np.ndarray] = []
    labels: List[int] = []

    for traj, label in trajs_labels:
        state = compute_memory(traj, K, r, lam, W_Theta, Q, solver="osqp")
        cloud_padded = pad_rows(state.point_cloud, K)
        cloud_flat = cloud_padded.flatten().astype(np.float32)
        topo = summarize_diagrams(state.persistence_diagrams)
        feat = np.concatenate([cloud_flat, topo])
        features.append(feat)
        labels.append(label)

    X = np.stack(features).astype(np.float64)
    y = np.array(labels, dtype=np.int64)

    # Simple train/test split
    n = len(X)
    n_train = max(4, int(0.7 * n))
    perm = rng.permutation(n)
    X_train, y_train = X[perm[:n_train]], y[perm[:n_train]]
    X_test, y_test = X[perm[n_train:]], y[perm[n_train:]]

    return ridge_probe_accuracy(X_train, y_train, X_test, y_test)


# ── Single-Seed Experiment ─────────────────────────────────────────────────

def run_single_seed(config: Any, seed: int) -> Dict[str, float]:
    """Run EMP-08 for a single seed."""
    if pl is not None:
        pl.seed_everything(seed, workers=True)
    rng = np.random.default_rng(seed)
    cfg = config

    d = cfg.trajectory.d
    T = cfg.trajectory.T
    k = cfg.memory.k
    Q = cfg.memory.Q
    n_trials = getattr(cfg.data, "n_trials", 30)
    tolerance = getattr(cfg, "tolerance", 2)

    sweep = cfg.sweep if hasattr(cfg, "sweep") else cfg
    lambda_values = getattr(sweep, "lambda_values", [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
    K_values = getattr(sweep, "K_values", [3, 5, 10, 15, 20, 30])
    r_values = getattr(sweep, "r_values", [0, 1, 2, 3, 5, 8])

    # Fixed defaults for single-axis sweeps
    K_default = cfg.memory.K
    r_default = cfg.memory.r
    lam_default = cfg.memory.lam

    results: Dict[str, float] = {}

    # ── Test A: Lambda Sensitivity (fixed K, r) ───────────────────────
    for lam_val in lambda_values:
        f1_list: List[float] = []
        l2_list: List[float] = []
        for _ in range(n_trials):
            traj, gt = _generate_labeled_trajectory(d, T, rng)
            metrics = _evaluate_recovery(traj, gt, K_default, r_default, lam_val, tolerance)
            f1_list.append(metrics["f1"])
            l2_list.append(metrics["y_l2_norm"])

        results[f"A_lam{lam_val}_f1_mean"] = float(np.mean(f1_list))
        results[f"A_lam{lam_val}_f1_std"] = float(np.std(f1_list, ddof=1)) if len(f1_list) > 1 else 0.0
        results[f"A_lam{lam_val}_l2_mean"] = float(np.mean(l2_list))

    # ── Test B: K Sensitivity (fixed lambda, r) ──────────────────────
    for K_val in K_values:
        f1_list = []
        det_list: List[float] = []
        for _ in range(n_trials):
            traj, gt = _generate_labeled_trajectory(d, T, rng)
            metrics = _evaluate_recovery(traj, gt, K_val, r_default, lam_default, tolerance)
            f1_list.append(metrics["f1"])
            det_list.append(metrics["n_detected"])

        results[f"B_K{K_val}_f1_mean"] = float(np.mean(f1_list))
        results[f"B_K{K_val}_f1_std"] = float(np.std(f1_list, ddof=1)) if len(f1_list) > 1 else 0.0
        results[f"B_K{K_val}_n_detected_mean"] = float(np.mean(det_list))

    from experiments.empirical.common.tasks import generate_memory_task_dataset
    test_data = generate_memory_task_dataset(rng, "delayed_retrieval", 60, T, d, 0.25)
    save_experiment_npz("EMP-08", seed, {
        "test_sequences": np.stack([s.sequence for s in test_data]),
        "test_labels": np.array([s.label for s in test_data])
    }, cfg.output_dir)

    # ── Test C: r Sensitivity (fixed lambda, K) ──────────────────────
    for r_val in r_values:
        f1_list = []
        sparsity_list: List[float] = []
        for _ in range(n_trials):
            traj, gt = _generate_labeled_trajectory(d, T, rng)
            metrics = _evaluate_recovery(traj, gt, K_default, r_val, lam_default, tolerance)
            f1_list.append(metrics["f1"])
            sparsity_list.append(metrics["y_sparsity"])

        results[f"C_r{r_val}_f1_mean"] = float(np.mean(f1_list))
        results[f"C_r{r_val}_f1_std"] = float(np.std(f1_list, ddof=1)) if len(f1_list) > 1 else 0.0
        results[f"C_r{r_val}_sparsity_mean"] = float(np.mean(sparsity_list))

    # ── Test D: 2D Heatmap — Lambda × K joint sweep ──────────────────
    # Subset for computational feasibility
    lam_heat = getattr(sweep, "heatmap_lambdas", [0.1, 0.5, 1.0, 5.0, 10.0])
    K_heat = getattr(sweep, "heatmap_Ks", [5, 10, 20])

    for lam_val in lam_heat:
        for K_val in K_heat:
            f1_list = []
            for _ in range(n_trials):
                traj, gt = _generate_labeled_trajectory(d, T, rng)
                metrics = _evaluate_recovery(traj, gt, K_val, r_default, lam_val, tolerance)
                f1_list.append(metrics["f1"])
            results[f"D_lam{lam_val}_K{K_val}_f1"] = float(np.mean(f1_list))

    # ── Test E: Downstream task sensitivity ───────────────────────────
    # Quick probe accuracy at different (K, lam) combos
    from experiments.empirical.common.tasks import generate_memory_task_dataset
    task_data = generate_memory_task_dataset(rng, "delayed_retrieval", 60, T, d, 0.25)
    trajs_labels = [(s.sequence.astype(np.float64), s.label) for s in task_data]

    for K_val in [5, 10, 20]:
        for lam_val in [0.1, 1.0, 10.0]:
            acc = _ridge_probe_from_memory(trajs_labels, K_val, r_default, lam_val, k, Q, rng)
            results[f"E_K{K_val}_lam{lam_val}_probe_acc"] = acc

    return results


# ── Entry Point ────────────────────────────────────────────────────────────

def run_experiment(config: Any = None) -> Dict[str, Any]:
    """Run EMP-08 across all seeds and return aggregated report."""
    if config is None:
        config = load_emp_config("EMP-08")

    report = run_multi_seed(
        experiment_fn=run_single_seed,
        config=config,
        seeds=config.training.seeds,
        experiment_id="EMP-08",
        output_dir=config.output_dir,
    )

    agg = report["aggregated"]

    # Pass criterion: best F1 across lambda sweep should be > 0.5
    # AND performance should not collapse entirely at any single point
    best_lam_f1 = max(
        agg.get(k, {}).get("mean", 0.0)
        for k in agg if k.startswith("A_lam") and k.endswith("_f1_mean")
    ) if any(k.startswith("A_lam") for k in agg) else 0.0

    best_K_f1 = max(
        agg.get(k, {}).get("mean", 0.0)
        for k in agg if k.startswith("B_K") and k.endswith("_f1_mean")
    ) if any(k.startswith("B_K") for k in agg) else 0.0

    # Check that >50% of lambda values give F1 > 0.3 (robustness)
    sweep = config.sweep if hasattr(config, "sweep") else config
    lambda_values = getattr(sweep, "lambda_values", [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
    robust_count = sum(
        1 for lam in lambda_values
        if agg.get(f"A_lam{lam}_f1_mean", {}).get("mean", 0.0) > 0.3
    )
    robust_ratio = robust_count / max(len(lambda_values), 1)

    passed = best_lam_f1 > 0.5 and best_K_f1 > 0.5 and robust_ratio > 0.5

    report["experiment_name"] = "Hyperparameter Sensitivity"
    report["pass_criterion"] = (
        f"best_lam_f1>{0.5} AND best_K_f1>{0.5} AND robust_ratio>{0.5}"
    )
    report["passed"] = passed
    report["best_lam_f1"] = best_lam_f1
    report["best_K_f1"] = best_K_f1
    report["robust_ratio"] = robust_ratio

    log.info("[EMP-08] best_lam_f1=%.4f  best_K_f1=%.4f  robust=%.0f%%  passed=%s",
             best_lam_f1, best_K_f1, robust_ratio * 100, passed)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if str(Path(__file__).resolve().parent.parent.parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
    result = run_experiment()
    print(f"\nEMP-08 PASSED: {result['passed']}")
