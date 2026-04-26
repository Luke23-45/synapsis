"""
EMP-12: Exact Recovery Verification

Empirically validates Theorem 6.3 — exact recovery of the intended anchor
set under the strict dominance condition.

Formal Claim (Thm 6.3):
    Let C = {c_1 < ... < c_m} ⊆ {2,...,T} satisfy:
        (i)   m ≤ K
        (ii)  c_{j+1} − c_j > r  for all j
        (iii) min_{t ∈ C} y*_t  >  max_{u ∉ C} y*_u   ("strict dominance")
    Then I*(x_{1:T}) = C.

Z2 Reference: §6 of 02_rigorous_architecture.md, Thm 6.3
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

log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _generate_trajectory_with_known_events(
    d: int, T: int, event_positions: List[int],
    event_magnitude: float, noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a trajectory with strong transitions at specified positions."""
    traj = np.cumsum(rng.standard_normal((T, d)) * noise_std, axis=0)
    for pos in event_positions:
        if 1 <= pos < T:
            direction = rng.standard_normal(d)
            direction /= np.linalg.norm(direction) + 1e-12
            traj[pos:] += direction * event_magnitude
    return traj.astype(np.float64)


def _sample_event_set(
    K: int, r: int, T: int, rng: np.random.Generator,
) -> List[int]:
    """Sample a random admissible event set C with |C| ∈ [1, K] and spacing > r."""
    max_m = min(K, (T - 2) // (r + 1))
    if max_m < 1:
        return []
    m = int(rng.integers(1, max_m + 1))

    C: List[int] = []
    current = int(rng.integers(2, max(3, min(10, T // (m + 1)))))
    for _ in range(m):
        if current >= T:
            break
        C.append(current)
        current += r + 1 + int(rng.integers(1, 5))
    return C


def _dominance_margin(y_star: np.ndarray, C: List[int], K: int) -> float:
    """margin = min_{t∈C} y*_t − max_{u∉C, u>0} y*_u.  Positive ⇒ strict dominance.

    Fully vectorised via np.isin — no Python-level iteration.
    Enforces Theorem 6.3 second condition: if m < K, max_{u∉C} y*_u must be <= 0.
    """
    if not C:
        return 0.0
    C_arr = np.asarray(C, dtype=np.intp)
    min_c = float(y_star[C_arr].min())

    # All positions ≥ 1 that are NOT in C — ultra-fast boolean masking
    mask = np.ones(len(y_star), dtype=bool)
    mask[0] = False
    mask[C_arr] = False
    non_c_vals = y_star[mask]
    max_nc = float(non_c_vals.max()) if len(non_c_vals) > 0 else 0.0
    
    if len(C) < K and max_nc > 0.0:
        return -1.0
        
    return min_c - max_nc


def _exact_recovery(I_star: List[int], C: List[int]) -> bool:
    return sorted(I_star) == sorted(C)


def _recovery_f1(I_star: List[int], C: List[int]) -> float:
    if not C and not I_star:
        return 1.0
    if not C or not I_star:
        return 0.0
    tp = len(set(I_star) & set(C))
    prec = tp / len(I_star)
    rec = tp / len(C)
    return 2.0 * prec * rec / (prec + rec) if (prec + rec) > 1e-10 else 0.0


def _run_pipeline(traj: np.ndarray, K: int, r: int, lam: float):
    """Score → saliency → selector → projection.  Returns (y*, I*)."""
    scores = sharp_event_score(traj)
    sal = normalize_saliency(scores, mode="identity")
    y_star = solve_relaxed_selector(sal, K, r, lam, solver="osqp")
    I_star = hard_projection(y_star, K, r)
    return y_star, I_star


# ── Single-Seed Experiment ─────────────────────────────────────────────────

def run_single_seed(config: Any, seed: int) -> Dict[str, float]:
    """Run EMP-12 for a single seed."""
    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed, workers=True)
    except ImportError:
        pass
    rng = np.random.default_rng(seed)
    cfg = config

    d = cfg.trajectory.d
    T = cfg.trajectory.T
    K = cfg.memory.K
    r = cfg.memory.r

    sweep = cfg.sweep if hasattr(cfg, "sweep") else cfg
    n_trials = getattr(sweep, "n_trials", 30)
    lambda_values = getattr(sweep, "lambda_values",
                            [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
    event_magnitudes = getattr(sweep, "event_magnitudes", [3.0, 5.0, 10.0])
    noise_stds = getattr(sweep, "noise_stds", [0.01, 0.05, 0.1])

    results: Dict[str, float] = {}

    # ── Test A: Flat sweep over (trial, mag, noise, λ) ────────────────
    dom_ok = 0
    dom_total = 0
    nodom_ok = 0
    nodom_total = 0
    all_margins: List[float] = []
    trajs: List[np.ndarray] = []

    configs_A = [
        (trial, mag, noise, lam)
        for trial in range(n_trials)
        for mag in event_magnitudes
        for noise in noise_stds
        for lam in lambda_values
    ]
    log.info("[EMP-12] Test A: %d (trial×mag×noise×λ) configs", len(configs_A))

    for trial, mag, noise, lam in configs_A:
        C = _sample_event_set(K, r, T, rng)
        if not C:
            continue

        traj = _generate_trajectory_with_known_events(d, T, C, mag, noise, rng)
        trajs.append(traj)
        y_star, I_star = _run_pipeline(traj, K, r, lam)
        margin = _dominance_margin(y_star, C, K)
        all_margins.append(margin)

        if margin > 0:
            dom_total += 1
            if _exact_recovery(I_star, C):
                dom_ok += 1
        else:
            nodom_total += 1
            if _exact_recovery(I_star, C):
                nodom_ok += 1

    results["A_dominance_recovery_rate"] = dom_ok / max(dom_total, 1)
    results["A_dominance_total"] = float(dom_total)
    results["A_non_dominance_recovery_rate"] = nodom_ok / max(nodom_total, 1)
    results["A_non_dominance_total"] = float(nodom_total)

    if all_margins:
        margins_arr = np.asarray(all_margins)
        results["A_mean_margin"] = float(margins_arr.mean())
        results["A_positive_margin_frac"] = float((margins_arr > 0).mean())
    else:
        results["A_mean_margin"] = 0.0
        results["A_positive_margin_frac"] = 0.0

    save_experiment_npz("EMP-12", seed, {"test_trajectories": np.stack(trajs) if trajs else np.zeros(0)}, cfg.output_dir)

    # ── Test B: Per-λ Margin and F1 ───────────────────────────────────
    for lam in lambda_values:
        margins: List[float] = []
        f1s: List[float] = []
        exacts: List[float] = []

        for trial in range(n_trials):
            C = _sample_event_set(K, r, T, rng)
            if not C:
                continue
            traj = _generate_trajectory_with_known_events(d, T, C, 5.0, 0.05, rng)
            y_star, I_star = _run_pipeline(traj, K, r, lam)
            margins.append(_dominance_margin(y_star, C, K))
            f1s.append(_recovery_f1(I_star, C))
            exacts.append(1.0 if _exact_recovery(I_star, C) else 0.0)

        if margins:
            results[f"B_lam{lam}_mean_margin"] = float(np.mean(margins))
            results[f"B_lam{lam}_mean_f1"] = float(np.mean(f1s))
            results[f"B_lam{lam}_exact_rate"] = float(np.mean(exacts))
        else:
            results[f"B_lam{lam}_mean_margin"] = 0.0
            results[f"B_lam{lam}_mean_f1"] = 0.0
            results[f"B_lam{lam}_exact_rate"] = 0.0

    # ── Test C: Edge Cases ────────────────────────────────────────────
    edge_pass = 0
    edge_total = 0

    # C1: m = K (saturated budget)
    C_sat: List[int] = []
    pos = 2
    for _ in range(K):
        if pos >= T:
            break
        C_sat.append(pos)
        pos += r + 2
    if len(C_sat) == K:
        edge_total += 1
        traj = _generate_trajectory_with_known_events(d, T, C_sat, 10.0, 0.01, rng)
        y_star, I_star = _run_pipeline(traj, K, r, 0.01)
        margin = _dominance_margin(y_star, C_sat, K)
        if margin > 0 and _exact_recovery(I_star, C_sat):
            edge_pass += 1
        results["C_saturated_margin"] = margin

    # C2: m = 1 (single event)
    edge_total += 1
    traj = _generate_trajectory_with_known_events(d, T, [T // 2], 10.0, 0.01, rng)
    y_star, I_star = _run_pipeline(traj, K, r, 0.01)
    margin = _dominance_margin(y_star, [T // 2], K)
    if margin > 0 and _exact_recovery(I_star, [T // 2]):
        edge_pass += 1
    results["C_single_margin"] = margin

    # C3: Events at boundaries
    C_bnd = [1, T - 2]
    if T - 2 - 1 > r:
        edge_total += 1
        traj = _generate_trajectory_with_known_events(d, T, C_bnd, 10.0, 0.01, rng)
        y_star, I_star = _run_pipeline(traj, K, r, 0.01)
        margin = _dominance_margin(y_star, C_bnd, K)
        if margin > 0 and _exact_recovery(I_star, C_bnd):
            edge_pass += 1
        results["C_boundary_margin"] = margin

    results["C_edge_pass_rate"] = edge_pass / max(edge_total, 1)

    return results


# ── Entry Point ────────────────────────────────────────────────────────────

def run_experiment(config: Any = None) -> Dict[str, Any]:
    """Run EMP-12 across all seeds and return aggregated report."""
    if config is None:
        config = load_emp_config("EMP-12")

    report = run_multi_seed(
        experiment_fn=run_single_seed,
        config=config,
        seeds=config.training.seeds,
        experiment_id="EMP-12",
        output_dir=config.output_dir,
    )

    agg = report["aggregated"]
    dom_rate = agg.get("A_dominance_recovery_rate", {}).get("mean", 0.0)
    dom_total = agg.get("A_dominance_total", {}).get("mean", 0.0)
    has_enough = dom_total >= 10.0

    passed = dom_rate >= 0.95 and has_enough

    report["experiment_name"] = "Exact Recovery Verification"
    report["pass_criterion"] = (
        f"dominance_recovery >= 0.95 (obs: {dom_rate:.4f}) "
        f"AND dominance_total >= 10 (obs: {dom_total:.0f})"
    )
    report["passed"] = passed

    log.info("[EMP-12] dom_rate=%.4f dom_n=%.0f passed=%s", dom_rate, dom_total, passed)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if str(Path(__file__).resolve().parent.parent.parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
    result = run_experiment()
    print(f"\nEMP-12 PASSED: {result['passed']}")
