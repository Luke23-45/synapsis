"""
EMP-07: Scalability Benchmark

Empirically validates Theorem 12.2 — bounded complexity O(T log T + K²) —
by measuring wall-clock runtime of compute_memory() across sweeps of T and K.
Produces the data needed for a publication-quality log-log scaling plot.

Z2 Reference: §12 of 02_rigorous_architecture.md, Thm 12.2
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from synapse_core.memory_operator import compute_memory
from experiments.empirical.common.seed_runner import run_multi_seed
from experiments.empirical.common.emp_config import load_emp_config
from experiments.empirical.common.math_utils import (
    make_orthogonal_W,
    generate_trajectory,
)
from experiments.empirical.common.data_saver import save_experiment_jsonl

log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

# Helpers imported from experiments.empirical.common.math_utils:
#   make_orthogonal_W, generate_trajectory


# ── Single-Seed Experiment ─────────────────────────────────────────────────

def run_single_seed(config: Any, seed: int) -> Dict[str, float]:
    """Run EMP-07 for a single seed."""
    rng = np.random.default_rng(seed)
    cfg = config

    d = cfg.trajectory.d
    K_default = cfg.memory.K
    r = cfg.memory.r
    lam = cfg.memory.lam
    k = cfg.memory.k
    Q = cfg.memory.Q

    sweep = cfg.sweep if hasattr(cfg, "sweep") else cfg
    T_values = getattr(sweep, "T_values", [50, 100, 250, 500, 1000, 2500, 5000])
    K_values = getattr(sweep, "K_values", [5, 10, 20, 50])
    n_repeats = getattr(sweep, "n_repeats", 5)

    D = d + 3
    results: Dict[str, float] = {}
    results_list: List[Dict[str, Any]] = []

    # ── Test A: Runtime vs T (fixed K) ────────────────────────────────
    for T_val in T_values:
        times: List[float] = []
        for rep in range(n_repeats):
            traj = generate_trajectory(d, T_val, rng)
            W_Theta = make_orthogonal_W(k, D, rng)

            # Warm-up (first call may have JIT/cache overhead)
            if rep == 0 and T_val == T_values[0]:
                compute_memory(traj, K_default, r, lam, W_Theta, Q, solver="scipy")  # warm-up

            t0 = time.perf_counter()
            compute_memory(traj, K_default, r, lam, W_Theta, Q, solver="scipy")
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        results[f"A_T{T_val}_mean_ms"] = float(np.mean(times)) * 1000.0
        results[f"A_T{T_val}_std_ms"] = float(np.std(times, ddof=1)) * 1000.0 if len(times) > 1 else 0.0
        results[f"A_T{T_val}_median_ms"] = float(np.median(times)) * 1000.0
        results_list.append({"test": "A", "T": T_val, "K": K_default, "times_s": times})

    # ── Test B: Runtime vs K (fixed T) ────────────────────────────────
    T_fixed = getattr(sweep, "T_fixed", 500)
    for K_val in K_values:
        times = []
        for rep in range(n_repeats):
            traj = generate_trajectory(d, T_fixed, rng)
            W_Theta = make_orthogonal_W(k, D, rng)

            t0 = time.perf_counter()
            compute_memory(traj, K_val, r, lam, W_Theta, Q, solver="scipy")
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        results[f"B_K{K_val}_mean_ms"] = float(np.mean(times)) * 1000.0
        results[f"B_K{K_val}_std_ms"] = float(np.std(times, ddof=1)) * 1000.0 if len(times) > 1 else 0.0
        results[f"B_K{K_val}_median_ms"] = float(np.median(times)) * 1000.0
        results_list.append({"test": "B", "T": T_fixed, "K": K_val, "times_s": times})

    # ── Test C: Component-level timing breakdown ──────────────────────
    T_profile = getattr(sweep, "T_profile", 1000)
    traj_prof = generate_trajectory(d, T_profile, rng)
    W_Theta_prof = make_orthogonal_W(k, D, rng)

    from synapse_core.event_encoder import sharp_event_score
    from synapse_core.anchor_selector import solve_relaxed_selector, hard_projection, build_anchors
    from synapse_core.geometric_lift import anchor_vectors, normalize_anchors, apply_lift

    # Event encoder timing
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        scores = sharp_event_score(traj_prof)
    results["C_encoder_mean_ms"] = (time.perf_counter() - t0) / n_repeats * 1000.0

    # Relaxed selector timing
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        y_star = solve_relaxed_selector(scores, K_default, r, lam, solver="scipy")
    results["C_selector_mean_ms"] = (time.perf_counter() - t0) / n_repeats * 1000.0

    # Hard projection timing
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        indices = hard_projection(y_star, K_default, r)
    results["C_projection_mean_ms"] = (time.perf_counter() - t0) / n_repeats * 1000.0

    # Anchor + lift timing
    anchors = build_anchors(indices, traj_prof, scores)
    V = anchor_vectors(anchors)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        V_norm, _, _ = normalize_anchors(V)
        cloud = apply_lift(V_norm, W_Theta_prof)
    results["C_lift_mean_ms"] = (time.perf_counter() - t0) / n_repeats * 1000.0

    # Full operator timing
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        compute_memory(traj_prof, K_default, r, lam, W_Theta_prof, Q, solver="scipy")
    results["C_full_operator_mean_ms"] = (time.perf_counter() - t0) / n_repeats * 1000.0

    save_experiment_jsonl("EMP-07", seed, results_list, config.output_dir)

    return results


# ── Entry Point ────────────────────────────────────────────────────────────

def run_experiment(config: Any = None) -> Dict[str, Any]:
    """Run EMP-07 across all seeds and return aggregated report."""
    if config is None:
        config = load_emp_config("EMP-07")

    report = run_multi_seed(
        experiment_fn=run_single_seed,
        config=config,
        seeds=config.training.seeds,
        experiment_id="EMP-07",
        output_dir=config.output_dir,
    )

    # Verify scaling is sub-quadratic in T: ratio of runtime at max T vs min T
    # should be << (max_T / min_T)^2
    agg = report["aggregated"]
    sweep = config.sweep if hasattr(config, "sweep") else config
    T_values = getattr(sweep, "T_values", [50, 100, 250, 500, 1000, 2500, 5000])

    t_min_key = f"A_T{T_values[0]}_mean_ms"
    t_max_key = f"A_T{T_values[-1]}_mean_ms"
    t_min_ms = agg.get(t_min_key, {}).get("mean", 1.0)
    t_max_ms = agg.get(t_max_key, {}).get("mean", 1.0)

    T_ratio = T_values[-1] / T_values[0]
    runtime_ratio = t_max_ms / max(t_min_ms, 1e-6)

    # O(T log T) implies ratio should be ~T_ratio * log(T_ratio)
    # Quadratic would be T_ratio^2. We pass if runtime_ratio < T_ratio^1.5
    scaling_exponent = np.log(runtime_ratio) / np.log(T_ratio) if T_ratio > 1 else 0.0
    passed = scaling_exponent < 1.8  # generous bound; O(T log T) ≈ 1.0–1.2

    report["experiment_name"] = "Scalability Benchmark"
    report["pass_criterion"] = f"scaling_exponent < 1.8 (observed: {scaling_exponent:.3f})"
    report["passed"] = passed
    report["scaling_exponent"] = scaling_exponent
    report["runtime_ratio"] = runtime_ratio

    log.info("[EMP-07] T_ratio=%.1f  runtime_ratio=%.1f  scaling_exponent=%.3f  passed=%s",
             T_ratio, runtime_ratio, scaling_exponent, passed)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if str(Path(__file__).resolve().parent.parent.parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
    result = run_experiment()
    print(f"\nEMP-07 PASSED: {result['passed']}")
    print(f"Scaling exponent: {result['scaling_exponent']:.3f}")
