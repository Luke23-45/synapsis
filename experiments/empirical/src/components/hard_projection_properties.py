"""
EMP-13: Hard Projection Properties

Empirically validates Proposition 6.1 (determinism & budget bound) and
Proposition 6.2 (refractory separation guarantee) for Proj_{K,r,T}.

Formal Claims:
    Prop 6.1:  I* is uniquely defined (deterministic) and |I*| ≤ K.
    Prop 6.2:  |i − j| > r  for all distinct i, j ∈ I*.

Z2 Reference: §6 of 02_rigorous_architecture.md, Prop 6.1, Prop 6.2
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from synapse_core.event_encoder import sharp_event_score
from synapse_core.saliency_normalizer import normalize_saliency
from synapse_core.anchor_selector import solve_relaxed_selector, hard_projection
from experiments.empirical.common.seed_runner import run_multi_seed
from experiments.empirical.common.emp_config import load_emp_config
from experiments.empirical.common.data_saver import save_experiment_npz
from experiments.empirical.common.math_utils import generate_trajectory

log = logging.getLogger(__name__)


# ── Structural Checks (vectorised where possible) ─────────────────────────

def _check_all_properties(
    I_star: List[int], K: int, r: int, y_star: np.ndarray,
) -> Dict[str, bool]:
    """Run all structural checks in one pass — no redundant iteration."""
    sorted_I = sorted(I_star)

    budget_ok = len(I_star) <= K
    sorted_ok = I_star == sorted_I
    no_zero_ok = 0 not in I_star

    # Separation: vectorised gap check
    if len(sorted_I) >= 2:
        gaps = np.diff(sorted_I)
        separation_ok = bool(np.all(gaps > r))
    else:
        separation_ok = True

    # Positive support: vectorised
    if I_star:
        support_ok = bool(np.all(y_star[np.array(I_star)] > 0))
    else:
        support_ok = True

    return {
        "budget": budget_ok,
        "separation": separation_ok,
        "sorted": sorted_ok,
        "no_zero": no_zero_ok,
        "support": support_ok,
    }


# ── Single-Seed Experiment ─────────────────────────────────────────────────

def run_single_seed(config: Any, seed: int) -> Dict[str, float]:
    """Run EMP-13 for a single seed."""
    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed, workers=True)
    except ImportError:
        pass
    rng = np.random.default_rng(seed)
    cfg = config

    d = cfg.trajectory.d
    T = cfg.trajectory.T

    sweep = cfg.sweep if hasattr(cfg, "sweep") else cfg
    K_values = getattr(sweep, "K_values", [1, 2, 5, 10, 20])
    r_values = getattr(sweep, "r_values", [0, 1, 2, 5, 10])
    lam_values = getattr(sweep, "lam_values", [0.01, 0.1, 1.0, 10.0])
    n_trials = getattr(sweep, "n_trials", 20)

    results: Dict[str, float] = {}

    # ── Test A: Determinism ───────────────────────────────────────────
    n_det_repeats = 10
    det_pass = 0
    det_total = 0

    for trial in range(n_trials):
        traj = generate_trajectory(d, T, rng)
        scores = sharp_event_score(traj)
        sal = normalize_saliency(scores, mode="identity")
        K = int(rng.choice(K_values))
        r = int(rng.choice(r_values))
        lam = float(rng.choice(lam_values))
        y_star = solve_relaxed_selector(sal, K, r, lam, solver="osqp")

        reference = hard_projection(y_star, K, r)
        det_total += 1
        if all(hard_projection(y_star, K, r) == reference for _ in range(n_det_repeats)):
            det_pass += 1

    results["A_determinism_pass_rate"] = det_pass / max(det_total, 1)

    # ── Test B+C: Budget + Separation (SINGLE pass) ──────────────────
    param_configs = [
        (K_val, r_val, lam_val, trial)
        for K_val in K_values
        for r_val in r_values
        for lam_val in lam_values
        for trial in range(n_trials)
    ]
    log.info("[EMP-13] Test B+C: %d (K×r×λ×trial) configs", len(param_configs))

    counters = {k: 0 for k in ["budget", "separation", "sorted", "no_zero", "support"]}
    bc_total = 0

    for K, r, lam, _trial in param_configs:
        traj = generate_trajectory(d, T, rng)
        scores = sharp_event_score(traj)
        sal = normalize_saliency(scores, mode="identity")
        y_star = solve_relaxed_selector(sal, K, r, lam, solver="osqp")
        I_star = hard_projection(y_star, K, r)

        checks = _check_all_properties(I_star, K, r, y_star)
        bc_total += 1
        for prop, ok in checks.items():
            if ok:
                counters[prop] += 1
            elif prop in ("budget", "separation"):
                log.warning(
                    "[EMP-13] %s violation: I*=%s K=%d r=%d lam=%.3f",
                    prop, sorted(I_star), K, r, lam,
                )

    for prop in counters:
        results[f"BC_{prop}_pass_rate"] = counters[prop] / max(bc_total, 1)
    results["BC_total"] = float(bc_total)

    # ── Test D: Adversarial Edge Cases ────────────────────────────────
    edge_pass = 0
    edge_total = 0

    d_configs = [
        (K_val, r_val)
        for K_val in [1, 5, 10]
        for r_val in [0, 2, 5]
    ]

    for K, r in d_configs:
        # D1: All y*_t equal (tie-breaking stress)
        edge_total += 1
        y_flat = np.full(T, 0.5, dtype=np.float64)
        y_flat[0] = 0.0
        I = hard_projection(y_flat, K, r)
        c = _check_all_properties(I, K, r, y_flat)
        if all(c.values()):
            edge_pass += 1

        # D2: Exactly K entries, spaced exactly r+1 apart
        edge_total += 1
        y_exact = np.zeros(T, dtype=np.float64)
        pos, cnt = 1, 0
        while cnt < K and pos < T:
            y_exact[pos] = 1.0 - cnt * 0.01
            pos += r + 1
            cnt += 1
        I = hard_projection(y_exact, K, r)
        c = _check_all_properties(I, K, r, y_exact)
        if all(c.values()):
            edge_pass += 1

        # D3: More than K positive entries
        edge_total += 1
        y_over = np.zeros(T, dtype=np.float64)
        n_pos = min(2 * K, T - 1)
        positions = rng.choice(range(1, T), size=n_pos, replace=False)
        y_over[positions] = rng.uniform(0.1, 1.0, size=n_pos)
        I = hard_projection(y_over, K, r)
        c = _check_all_properties(I, K, r, y_over)
        if all(c.values()):
            edge_pass += 1

        # D4: Empty support
        edge_total += 1
        y_zero = np.zeros(T, dtype=np.float64)
        I = hard_projection(y_zero, K, r)
        if len(I) == 0:
            edge_pass += 1

        # D5: Near-tolerance values
        edge_total += 1
        y_tiny = np.zeros(T, dtype=np.float64)
        y_tiny[T // 3] = 1e-5
        y_tiny[2 * T // 3] = 0.5
        I = hard_projection(y_tiny, K, r)
        c = _check_all_properties(I, K, r, y_tiny)
        if all(c.values()):
            edge_pass += 1

    results["D_edge_pass_rate"] = edge_pass / max(edge_total, 1)
    results["D_edge_total"] = float(edge_total)

    # ── Test E: Stress Test — Large Random y* ─────────────────────────
    T_stress_values = getattr(sweep, "T_stress_values", [100, 500, 1000])
    stress_configs = [
        (T_s, K_val, r_val, trial)
        for T_s in T_stress_values
        for K_val in [5, 20, 50]
        for r_val in [0, 2, 5]
        for trial in range(5)
    ]
    stress_pass = 0
    stress_total = len(stress_configs)

    for T_s, K, r, _trial in stress_configs:
        y_rand = np.zeros(T_s, dtype=np.float64)
        n_pos = rng.integers(1, min(3 * K, T_s - 1) + 1)
        pos = rng.choice(range(1, T_s), size=n_pos, replace=False)
        y_rand[pos] = rng.uniform(0.01, 1.0, size=n_pos)
        I = hard_projection(y_rand, K, r)
        c = _check_all_properties(I, K, r, y_rand)
        if all(c.values()):
            stress_pass += 1

    results["E_stress_pass_rate"] = stress_pass / max(stress_total, 1)

    save_experiment_npz("EMP-13", seed, {"last_stress_y": y_rand}, cfg.output_dir)

    return results


# ── Entry Point ────────────────────────────────────────────────────────────

def run_experiment(config: Any = None) -> Dict[str, Any]:
    """Run EMP-13 across all seeds and return aggregated report."""
    if config is None:
        config = load_emp_config("EMP-13")

    report = run_multi_seed(
        experiment_fn=run_single_seed,
        config=config,
        seeds=config.training.seeds,
        experiment_id="EMP-13",
        output_dir=config.output_dir,
    )

    agg = report["aggregated"]
    det = agg.get("A_determinism_pass_rate", {}).get("mean", 0.0)
    budget = agg.get("BC_budget_pass_rate", {}).get("mean", 0.0)
    sep = agg.get("BC_separation_pass_rate", {}).get("mean", 0.0)
    edge = agg.get("D_edge_pass_rate", {}).get("mean", 0.0)
    stress = agg.get("E_stress_pass_rate", {}).get("mean", 0.0)
    srt = agg.get("BC_sorted_pass_rate", {}).get("mean", 0.0)
    sup = agg.get("BC_support_pass_rate", {}).get("mean", 0.0)
    nz = agg.get("BC_no_zero_pass_rate", {}).get("mean", 0.0)

    passed = (
        det >= 1.0 - 1e-9
        and budget >= 1.0 - 1e-9
        and sep >= 1.0 - 1e-9
        and edge >= 0.95
        and stress >= 1.0 - 1e-9
        and srt >= 1.0 - 1e-9
        and sup >= 1.0 - 1e-9
        and nz >= 1.0 - 1e-9
    )

    report["experiment_name"] = "Hard Projection Properties"
    report["pass_criterion"] = (
        f"det={det:.4f} budget={budget:.4f} sep={sep:.4f} "
        f"edge={edge:.4f} stress={stress:.4f}"
    )
    report["passed"] = passed

    log.info(
        "[EMP-13] det=%.4f budget=%.4f sep=%.4f edge=%.4f "
        "stress=%.4f passed=%s",
        det, budget, sep, edge, stress, passed,
    )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if str(Path(__file__).resolve().parent.parent.parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
    result = run_experiment()
    print(f"\nEMP-13 PASSED: {result['passed']}")
