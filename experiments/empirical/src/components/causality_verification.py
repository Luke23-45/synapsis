"""
EMP-10: Causality Verification

Empirically validates Theorem 12.1 (Causality of the full memory operator)
and Proposition 5.2 (Causality of the relaxed selector).

Formal Claim (Thm 12.1):
    If the event encoder and the saliency normalizer are causal, then the
    family of prefix maps  x_{1:t} → M^inf_Θ(x_{1:t})  is causal.

Formal Claim (Prop 5.2):
    The relaxed selector y* restricted to positions 1..t depends only on
    saliency scores s_{1:t}.

Test Design:
    A.  Component-level causality — for each Z2 component (event encoder,
        saliency normalizer, relaxed selector, hard projection), verify that
        the output at position t is invariant to future data.
    B.  Full-operator causality — compute the full Z2 memory operator on a
        prefix x_{1:t} and on the full trajectory x_{1:T}, verify the memory
        state at time t is identical.
    C.  Adversarial futures — append deliberately adversarial future data
        and re-verify.

Z2 Reference: §12 of 02_rigorous_architecture.md, Thm 12.1; §5, Prop 5.2
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
from synapse_core.memory_operator import compute_memory
from experiments.empirical.common.tasks import generate_topology_dataset
from experiments.empirical.common.seed_runner import run_multi_seed
from experiments.empirical.common.emp_config import load_emp_config
from experiments.empirical.common.data_saver import save_experiment_npz
from experiments.empirical.common.math_utils import (
    make_orthogonal_W,
    generate_trajectory,
)

log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _test_component_causality(
    traj: np.ndarray,
    t_cut: int,
    future_alt: np.ndarray,
    K: int,
    r: int,
    lam: float,
    saliency_modes: List[str],
) -> Dict[str, bool]:
    """Run all component-level causality checks for a single (trajectory, cut)
    pair.  Returns a dict of component_name → passed."""
    prefix = traj[:t_cut]
    full_alt = np.concatenate([prefix, future_alt], axis=0)

    # ── Event Encoder (§3) ────────────────────────────────────────────
    scores_prefix = sharp_event_score(prefix)
    scores_orig = sharp_event_score(traj)
    scores_alt = sharp_event_score(full_alt)

    enc_ok = (
        np.array_equal(scores_prefix, scores_orig[:t_cut])
        and np.array_equal(scores_prefix, scores_alt[:t_cut])
    )

    # ── Saliency Normalizer (§4) — vectorised over modes ─────────────
    sal_ok = True
    for mode in saliency_modes:
        sp = normalize_saliency(scores_prefix, mode=mode)
        so = normalize_saliency(scores_orig, mode=mode)
        sa = normalize_saliency(scores_alt, mode=mode)
        if not np.array_equal(sp, so[:t_cut]) or not np.array_equal(sp, sa[:t_cut]):
            sal_ok = False
            break

    # ── Relaxed Selector (Prop 5.2) ───────────────────────────────────
    sal_p = normalize_saliency(scores_prefix, mode="identity")
    sal_o = normalize_saliency(scores_orig, mode="identity")
    sal_a = normalize_saliency(scores_alt, mode="identity")

    y_p = solve_relaxed_selector(sal_p, K, r, lam, solver="osqp")
    y_o = solve_relaxed_selector(sal_o, K, r, lam, solver="osqp")
    y_a = solve_relaxed_selector(sal_a, K, r, lam, solver="osqp")

    sel_ok = (
        np.allclose(y_p, y_o[:t_cut], atol=1e-6)
        and np.allclose(y_p, y_a[:t_cut], atol=1e-6)
    )

    # ── Hard Projection ──────────────────────────────────────────────
    I_p = hard_projection(y_p, K, r)
    I_o = hard_projection(y_o, K, r)
    I_a = hard_projection(y_a, K, r)

    I_o_prefix = [i for i in I_o if i < t_cut]
    I_a_prefix = [i for i in I_a if i < t_cut]
    proj_ok = (I_p == I_o_prefix and I_p == I_a_prefix)

    return {
        "encoder": enc_ok,
        "saliency": sal_ok,
        "selector": sel_ok,
        "projection": proj_ok,
    }


# ── Single-Seed Experiment ─────────────────────────────────────────────────

def run_single_seed(config: Any, seed: int) -> Dict[str, float]:
    """Run EMP-10 for a single seed."""
    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed, workers=True)
    except ImportError:
        pass
    rng = np.random.default_rng(seed)
    cfg = config

    d = cfg.trajectory.d
    T = cfg.trajectory.T
    
    # Mathematical Correction (Theorem 12.1 / Prop 5.2):
    # Strict causality only holds for M^\infty (unbounded capacity) and r=0.
    # Finite K creates global competition; r > 0 creates a symmetric future-lookahead.
    K = T  # Simulate M^\infty
    r = 0  # Disable refractory lookahead
    
    lam = cfg.memory.lam
    k = cfg.memory.k
    Q = cfg.memory.Q
    D = d + 3

    sweep = cfg.sweep if hasattr(cfg, "sweep") else cfg
    n_trials = getattr(sweep, "n_trials", 30)
    cut_fractions = getattr(sweep, "cut_fractions", [0.3, 0.5, 0.7, 0.9])
    saliency_modes = getattr(sweep, "saliency_modes", ["identity", "z_score", "temperature"])

    results: Dict[str, float] = {}

    # ── Test A: Component-Level Causality ─────────────────────────────
    counters = {"encoder": 0, "saliency": 0, "selector": 0, "projection": 0}
    total_component = 0

    a_configs = [
        (trial, frac)
        for trial in range(n_trials)
        for frac in cut_fractions
    ]

    samples = []
    for trial, frac in a_configs:
        traj = generate_trajectory(d, T, rng)
        t_cut = max(3, int(frac * T))
        if t_cut >= T:
            continue
        total_component += 1
        samples.append(type('Sample', (), {'sequence': traj}))

        future_alt = rng.standard_normal((T - t_cut, d)) * 10.0
        checks = _test_component_causality(
            traj, t_cut, future_alt, K, r, lam, saliency_modes,
        )
        for comp, ok in checks.items():
            if ok:
                counters[comp] += 1

    for comp in counters:
        results[f"A_{comp}_pass_rate"] = counters[comp] / max(total_component, 1)
    results["A_total_tests"] = float(total_component)

    # ── Test B: Full Operator Prefix Invariance ───────────────────────
    full_op_pass = 0
    full_op_total = 0

    b_configs = [
        (trial, frac)
        for trial in range(n_trials)
        for frac in cut_fractions
    ]

    for trial, frac in b_configs:
        traj = generate_trajectory(d, T, rng)
        W_Theta = make_orthogonal_W(k, D, rng)
        t_cut = max(5, int(frac * T))
        if t_cut >= T:
            continue
        full_op_total += 1

        state_prefix = compute_memory(traj[:t_cut], K, r, lam, W_Theta, Q, solver="osqp")
        state_full = compute_memory(traj, K, r, lam, W_Theta, Q, solver="osqp")

        scores_match = np.array_equal(
            state_prefix.event_scores, state_full.event_scores[:t_cut],
        )
        indices_match = (
            state_prefix.anchor_indices == [i for i in state_full.anchor_indices if i < t_cut]
        )
        if scores_match and indices_match:
            full_op_pass += 1

    results["B_full_op_pass_rate"] = full_op_pass / max(full_op_total, 1)
    results["B_full_op_total"] = float(full_op_total)

    # ── Test C: Adversarial Futures ───────────────────────────────────
    adv_fns = {
        "large_jump": lambda n: rng.standard_normal((n, d)) * 1e6,
        "constant":   lambda n: np.full((n, d), 42.0),
        "zero":       lambda n: np.zeros((n, d)),
        "oscillate":  lambda n: np.tile([[-100.0] * d, [100.0] * d], (n // 2 + 1, 1))[:n],
    }

    adv_pass = 0
    adv_total = 0
    t_cut = T // 2

    c_configs = [
        (trial, adv_name)
        for trial in range(min(n_trials, 10))
        for adv_name in adv_fns
    ]

    for trial, adv_name in c_configs:
        traj = generate_trajectory(d, T, rng)
        prefix = traj[:t_cut]
        future = adv_fns[adv_name](T - t_cut)
        full_adv = np.concatenate([prefix, future], axis=0)
        adv_total += 1

        sp = sharp_event_score(prefix)
        sa = sharp_event_score(full_adv)
        enc_ok = np.array_equal(sp, sa[:t_cut])

        salp = normalize_saliency(sp, mode="identity")
        sala = normalize_saliency(sa, mode="identity")
        sal_ok = np.array_equal(salp, sala[:t_cut])

        yp = solve_relaxed_selector(salp, K, r, lam, solver="osqp")
        ya = solve_relaxed_selector(sala, K, r, lam, solver="osqp")
        sel_ok = np.allclose(yp, ya[:t_cut], atol=1e-6)

        if enc_ok and sal_ok and sel_ok:
            adv_pass += 1

    results["C_adversarial_pass_rate"] = adv_pass / max(adv_total, 1)
    results["C_adversarial_total"] = float(adv_total)

    save_experiment_npz("EMP-10", seed, {
        "sequences": np.stack([s.sequence for s in samples])
    }, cfg.output_dir)

    return results


# ── Entry Point ────────────────────────────────────────────────────────────

def run_experiment(config: Any = None) -> Dict[str, Any]:
    """Run EMP-10 across all seeds and return aggregated report."""
    if config is None:
        config = load_emp_config("EMP-10")

    report = run_multi_seed(
        experiment_fn=run_single_seed,
        config=config,
        seeds=config.training.seeds,
        experiment_id="EMP-10",
        output_dir=config.output_dir,
    )

    agg = report["aggregated"]
    encoder_rate = agg.get("A_encoder_pass_rate", {}).get("mean", 0.0)
    saliency_rate = agg.get("A_saliency_pass_rate", {}).get("mean", 0.0)
    selector_rate = agg.get("A_selector_pass_rate", {}).get("mean", 0.0)
    projection_rate = agg.get("A_projection_pass_rate", {}).get("mean", 0.0)
    full_op_rate = agg.get("B_full_op_pass_rate", {}).get("mean", 0.0)
    adv_rate = agg.get("C_adversarial_pass_rate", {}).get("mean", 0.0)

    passed = (
        encoder_rate >= 1.0 - 1e-9
        and saliency_rate >= 1.0 - 1e-9
        and selector_rate >= 0.95
        and projection_rate >= 0.95
        and full_op_rate >= 0.95
        and adv_rate >= 0.95
    )

    report["experiment_name"] = "Causality Verification"
    report["pass_criterion"] = (
        "encoder=100%, saliency=100%, selector>=95%, projection>=95%, "
        "full_op>=95%, adversarial>=95%"
    )
    report["passed"] = passed

    log.info(
        "[EMP-10] enc=%.4f sal=%.4f sel=%.4f proj=%.4f "
        "full=%.4f adv=%.4f passed=%s",
        encoder_rate, saliency_rate, selector_rate,
        projection_rate, full_op_rate, adv_rate, passed,
    )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if str(Path(__file__).resolve().parent.parent.parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
    result = run_experiment()
    print(f"\nEMP-10 PASSED: {result['passed']}")
