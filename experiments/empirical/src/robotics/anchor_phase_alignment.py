"""
EZ2-01: Anchor Phase Alignment (Robotics) — Empirical Experiment.

Z2 Reference: §7–8 of 02_rigorous_architecture.md
Formal Claims: §7 (Anchor Sequence), §8 (Normalized Anchor Geometry)

Tests that Z2 anchors correspond to semantically important transitions
in robotics trajectories, and Z2 compression preserves more task-relevant
structure than naive baselines.

Modernized to use:
  - YAML config via load_emp_config("EZ2-01")
  - Multi-seed orchestration via run_multi_seed()
  - Shared match_f1 from common/metrics.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl

from synapse_core.memory_operator import compute_memory
from experiments.common.trajectory_generators import piecewise_constant
from experiments.empirical.common.metrics import match_f1
from experiments.empirical.common.seed_runner import run_multi_seed
from experiments.empirical.common.emp_config import load_emp_config
from experiments.empirical.common.data_saver import save_experiment_npz

log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_orthogonal_W(k: int, D: int, rng: np.random.Generator) -> np.ndarray:
    if k <= D:
        A = rng.standard_normal((D, D))
        Q, _ = np.linalg.qr(A)
        return Q[:k, :].astype(np.float64)
    W = np.zeros((k, D), dtype=np.float64)
    W[:D, :D] = np.eye(D)
    return W


def _generate_labeled_phase_trajectory(
    d: int, T: int, num_segments: int, seed: int,
) -> Tuple[np.ndarray, List[int]]:
    """Generate piecewise-constant trajectory with known phase boundaries."""
    n_cps = max(1, num_segments - 1)
    cps = list(np.linspace(T // (n_cps + 1), T - T // (n_cps + 1), num=n_cps, dtype=int))
    traj, gt_boundaries = piecewise_constant(d, T, cps, seed=seed)
    return traj, gt_boundaries


def _boundary_hit_rate(
    anchor_indices: List[int], boundaries: List[int], tolerance: int,
) -> float:
    """Fraction of phase boundaries hit by at least one anchor within tolerance."""
    if not boundaries:
        return 1.0
    hits = sum(1 for b in boundaries if any(abs(a - b) <= tolerance for a in anchor_indices))
    return hits / len(boundaries)


def _anchor_concentration(
    anchor_indices: List[int], boundaries: List[int], tolerance: int,
) -> float:
    """Fraction of anchors near at least one boundary (precision)."""
    if not anchor_indices:
        return 0.0
    near = sum(1 for a in anchor_indices if any(abs(a - b) <= tolerance for b in boundaries))
    return near / len(anchor_indices)


def _compute_f1(recall: float, precision: float) -> float:
    if precision + recall < 1e-10:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _topk_event_baseline(scores: np.ndarray, K: int, r: int) -> List[int]:
    order = sorted(range(1, len(scores)), key=lambda idx: (-scores[idx], idx))
    retained: List[int] = []
    for idx in order:
        if len(retained) >= K:
            break
        if all(abs(idx - prev) > r for prev in retained):
            retained.append(idx)
    retained.sort()
    return retained


# ── Single-Seed Experiment ─────────────────────────────────────────────────

def run_single_seed(config: Any, seed: int) -> Dict[str, float]:
    """Run EZ2-01 for a single seed."""
    pl.seed_everything(seed, workers=True)
    cfg = config
    rng = np.random.default_rng(seed)

    d = cfg.trajectory.d
    T = cfg.trajectory.T
    Q = cfg.memory.Q

    sweep = cfg.sweep if hasattr(cfg, "sweep") else cfg
    K_values = getattr(sweep, "K_values", [5, 10, 20])
    r_values = getattr(sweep, "r_values", [1, 2, 5])
    lam_values = getattr(sweep, "lam_values", [0.1, 0.5, 1.0, 5.0])
    k_values = getattr(sweep, "k_values", [4, 8])
    n_trials = getattr(cfg.data, "n_trials", 20)
    boundary_tolerance = 3
    if hasattr(cfg, "evaluation"):
        boundary_tolerance = getattr(cfg.evaluation, "boundary_tolerance", 3)

    D = d + 3
    results: Dict[str, float] = {}

    # ── Test A: Anchor-Boundary Alignment ─────────────────────────────
    best_z2_f1 = 0.0
    best_baseline_f1 = 0.0
    best_hit_rate = 0.0

    for K in K_values:
        for r in r_values:
            for lam in lam_values:
                for k in k_values:
                    z2_f1s: List[float] = []
                    uni_f1s: List[float] = []
                    topk_f1s: List[float] = []
                    hit_rates: List[float] = []

                    for trial in range(n_trials):
                        num_segments = max(3, min(K + 2, T // (r + 1)))
                        traj, gt_boundaries = _generate_labeled_phase_trajectory(
                            d, T, num_segments, int(rng.integers(2**31)),
                        )
                        W_Theta = _make_orthogonal_W(k, D, rng)
                        state = compute_memory(traj, K, r, lam, W_Theta, Q, solver="scipy")

                        hit_rate = _boundary_hit_rate(
                            state.anchor_indices, gt_boundaries, boundary_tolerance,
                        )
                        concentration = _anchor_concentration(
                            state.anchor_indices, gt_boundaries, boundary_tolerance,
                        )
                        z2_f1 = _compute_f1(hit_rate, concentration)

                        # Baselines
                        uniform_idx = list(np.linspace(1, T - 1, num=min(K, T - 1), dtype=int))
                        uni_hit = _boundary_hit_rate(uniform_idx, gt_boundaries, boundary_tolerance)
                        uni_conc = _anchor_concentration(uniform_idx, gt_boundaries, boundary_tolerance)
                        uni_f1 = _compute_f1(uni_hit, uni_conc)

                        topk_idx = _topk_event_baseline(state.event_scores, K, r)
                        topk_hit = _boundary_hit_rate(topk_idx, gt_boundaries, boundary_tolerance)
                        topk_conc = _anchor_concentration(topk_idx, gt_boundaries, boundary_tolerance)
                        topk_f1 = _compute_f1(topk_hit, topk_conc)

                        z2_f1s.append(z2_f1)
                        uni_f1s.append(uni_f1)
                        topk_f1s.append(topk_f1)
                        hit_rates.append(hit_rate)

                    mean_z2 = float(np.mean(z2_f1s))
                    mean_uni = float(np.mean(uni_f1s))
                    mean_topk = float(np.mean(topk_f1s))
                    mean_hit = float(np.mean(hit_rates))

                    best_z2_f1 = max(best_z2_f1, mean_z2)
                    best_baseline_f1 = max(best_baseline_f1, mean_uni, mean_topk)
                    best_hit_rate = max(best_hit_rate, mean_hit)

    results["A_best_z2_f1"] = best_z2_f1
    results["A_best_baseline_f1"] = best_baseline_f1
    results["A_best_hit_rate"] = best_hit_rate

    # ── Test B: Compression Retention ─────────────────────────────────
    for K in K_values:
        coverages: List[float] = []
        efficiencies: List[float] = []

        for trial in range(n_trials):
            r = 2
            lam = 1.0
            k = 8
            num_segments = max(3, min(K + 2, T // (r + 1)))
            traj, gt_boundaries = _generate_labeled_phase_trajectory(
                d, T, num_segments, int(rng.integers(2**31)),
            )
            W_Theta = _make_orthogonal_W(k, D, rng)
            state = compute_memory(traj, K, r, lam, W_Theta, Q, solver="scipy")
            coverage = _boundary_hit_rate(state.anchor_indices, gt_boundaries, boundary_tolerance)
            efficiency = len(state.anchors) / K if K > 0 else 0.0
            coverages.append(coverage)
            efficiencies.append(efficiency)

        results[f"B_K{K}_coverage"] = float(np.mean(coverages))
        results[f"B_K{K}_efficiency"] = float(np.mean(efficiencies))

    results["B_best_coverage"] = max(
        v for k, v in results.items() if k.startswith("B_") and k.endswith("_coverage")
    )

    # ── Test C: Normalization Effect ──────────────────────────────────
    for norm_mode in ["fit_from_data", "identity"]:
        f1_list: List[float] = []
        K, r, lam, k = 10, 2, 1.0, 8
        for trial in range(n_trials):
            traj, gt_boundaries = _generate_labeled_phase_trajectory(
                d, T, 8, int(rng.integers(2**31)),
            )
            W_Theta = _make_orthogonal_W(k, D, rng)
            if norm_mode == "identity":
                mu = np.zeros(D, dtype=np.float64)
                sigma = np.ones(D, dtype=np.float64)
            else:
                mu, sigma = None, None
            state = compute_memory(
                traj, K, r, lam, W_Theta, Q, mu=mu, sigma=sigma, solver="scipy",
            )
            hit = _boundary_hit_rate(state.anchor_indices, gt_boundaries, boundary_tolerance)
            conc = _anchor_concentration(state.anchor_indices, gt_boundaries, boundary_tolerance)
            f1_list.append(_compute_f1(hit, conc))
        results[f"C_{norm_mode}_f1"] = float(np.mean(f1_list))

    save_experiment_npz("EZ2-01", seed, {"last_traj": traj, "gt_boundaries": np.array(gt_boundaries)}, cfg.output_dir)

    return results


# ── Entry Point ────────────────────────────────────────────────────────────

def run_experiment(config: Any = None) -> Dict[str, Any]:
    """Run EZ2-01 across all seeds and return aggregated report."""
    if config is None:
        config = load_emp_config("EZ2-01")

    report = run_multi_seed(
        experiment_fn=run_single_seed,
        config=config,
        seeds=config.training.seeds,
        experiment_id="EZ2-01",
        output_dir=config.output_dir,
    )

    agg = report["aggregated"]
    z2_f1 = agg.get("A_best_z2_f1", {}).get("mean", 0.0)
    baseline_f1 = agg.get("A_best_baseline_f1", {}).get("mean", 0.0)
    hit_rate = agg.get("A_best_hit_rate", {}).get("mean", 0.0)
    coverage = agg.get("B_best_coverage", {}).get("mean", 0.0)

    hit_gate = 0.30
    cov_gate = 0.40
    if hasattr(config, "acceptance_gates"):
        hit_gate = getattr(config.acceptance_gates, "boundary_hit_rate", hit_gate)
        cov_gate = getattr(config.acceptance_gates, "compression_coverage", cov_gate)

    passed = (hit_rate >= hit_gate) and (z2_f1 >= baseline_f1) and (coverage >= cov_gate)

    report["experiment_name"] = "Anchor Phase Alignment"
    report["pass_criterion"] = (
        f"hit_rate >= {hit_gate} AND z2_f1 >= baseline_f1 AND coverage >= {cov_gate}"
    )
    report["passed"] = passed

    log.info("[EZ2-01] z2_f1=%.4f  baseline_f1=%.4f  hit=%.4f  coverage=%.4f  passed=%s",
             z2_f1, baseline_f1, hit_rate, coverage, passed)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if str(Path(__file__).resolve().parent.parent.parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
    result = run_experiment()
    print(f"\nEZ2-01 PASSED: {result['passed']}")
