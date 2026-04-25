"""
EZ2-03: Event-Sparse Recovery (Controlled) — Empirical Experiment.

Z2 Reference: §5–6 of 02_rigorous_architecture.md
Formal Claims: §5 (Relaxed Selector), §6 (Hard Projection)

Tests that Z2 anchor selection recovers salient events more robustly
than naive schemes under nuisance variation.

Modernized to use:
  - YAML config via load_emp_config("EZ2-03")
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

from synapse_core.anchor_selector import solve_relaxed_selector, hard_projection
from synapse_core.event_encoder import sharp_event_score
from experiments.common.trajectory_generators import piecewise_constant
from experiments.empirical.common.metrics import match_f1
from experiments.empirical.common.seed_runner import run_multi_seed
from experiments.empirical.common.emp_config import load_emp_config
from experiments.empirical.common.data_saver import save_experiment_npz

log = logging.getLogger(__name__)


# ── Data Generation ────────────────────────────────────────────────────────

def _generate_event_sparse(
    d: int, T: int, K: int, r: int, rng: np.random.Generator,
    min_gap: int = 0,
) -> Tuple[np.ndarray, List[int]]:
    """Generate a trajectory with sparse persistent transitions."""
    traj = np.zeros((T, d), dtype=np.float64)

    spacing = max(r + 1, min_gap)
    max_events = min(K, T // (spacing + 1))
    num_events = max(1, int(rng.integers(1, max_events + 1)))

    events: List[int] = []
    current = spacing
    for _ in range(num_events):
        if current >= T:
            break
        events.append(current)
        current += spacing + int(rng.integers(0, 3))

    level = np.zeros(d, dtype=np.float64)
    for e in events:
        level = level + rng.standard_normal(d) * 3.0
        traj[e:] += level

    traj += rng.standard_normal(traj.shape) * 0.05
    return traj, events


# ── Baselines ──────────────────────────────────────────────────────────────

def _topk_event_baseline(scores: np.ndarray, K: int, r: int) -> List[int]:
    """Top-K with refractory suppression."""
    order = sorted(range(1, len(scores)), key=lambda idx: (-scores[idx], idx))
    retained: List[int] = []
    for idx in order:
        if len(retained) >= K:
            break
        if all(abs(idx - prev) > r for prev in retained):
            retained.append(idx)
    retained.sort()
    return retained


def _peak_nms_baseline(scores: np.ndarray, K: int, r: int) -> List[int]:
    """Peak detection with non-maximum suppression."""
    peaks = [
        idx for idx in range(1, len(scores) - 1)
        if scores[idx] >= scores[idx - 1] and scores[idx] >= scores[idx + 1]
    ]
    order = sorted(peaks, key=lambda idx: (-scores[idx], idx))
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
    """Run EZ2-03 for a single seed."""
    pl.seed_everything(seed, workers=True)
    cfg = config
    rng = np.random.default_rng(seed)

    # Extract config with safe defaults
    sweep = cfg.sweep if hasattr(cfg, "sweep") else cfg
    noise_levels = getattr(sweep, "noise_levels", [0.05, 0.15, 0.3])
    distractor_levels = getattr(sweep, "distractor_levels", [0.2, 0.6, 1.2])
    lambda_values = getattr(sweep, "lambda_values", [0.01, 0.1, 0.5, 1.0, 5.0, 10.0])
    r_values = getattr(sweep, "r_values", [0, 1, 2, 5])

    n_trials = getattr(cfg.data, "n_trials", 30)
    tolerance = getattr(cfg.evaluation, "tolerance", 2) if hasattr(cfg, "evaluation") else 2
    T = cfg.trajectory.T
    d = cfg.trajectory.d
    default_K = cfg.memory.K
    default_r = cfg.memory.r
    default_lam = cfg.memory.lam

    results: Dict[str, float] = {}

    # ── Test A: Recovery Under Noise ──────────────────────────────────
    z2_f1_all: List[float] = []
    uniform_f1_all: List[float] = []
    topk_f1_all: List[float] = []
    peak_f1_all: List[float] = []

    for noise_std in noise_levels:
        for distractor_scale in distractor_levels:
            z2_f1s: List[float] = []
            uni_f1s: List[float] = []
            topk_f1s: List[float] = []
            peak_f1s: List[float] = []

            for trial in range(n_trials):
                traj, gt_events = _generate_event_sparse(
                    d, T, default_K, default_r, rng,
                )
                noisy_traj = traj + rng.standard_normal(traj.shape) * noise_std

                # Add distractors
                non_event = [t for t in range(1, T) if t not in gt_events]
                n_dist = min(5, len(non_event))
                if n_dist > 0:
                    distractor_locs = rng.choice(non_event, size=n_dist, replace=False)
                    for loc in distractor_locs:
                        noisy_traj[loc] += rng.standard_normal(d) * distractor_scale

                scores = sharp_event_score(noisy_traj)

                # Z2 method
                y_star = solve_relaxed_selector(scores, default_K, default_r, default_lam, solver="osqp")
                I_star = hard_projection(y_star, default_K, default_r)
                z2_f1s.append(match_f1(I_star, gt_events, tolerance))

                # Uniform baseline
                uniform_idx = list(np.linspace(1, T - 1, min(default_K, T - 1), dtype=int))
                uni_f1s.append(match_f1(uniform_idx, gt_events, tolerance))

                # Top-K baseline
                topk_idx = _topk_event_baseline(scores, default_K, default_r)
                topk_f1s.append(match_f1(topk_idx, gt_events, tolerance))

                # Peak NMS baseline
                peak_idx = _peak_nms_baseline(scores, default_K, default_r)
                peak_f1s.append(match_f1(peak_idx, gt_events, tolerance))

            key = f"n{noise_std}_d{distractor_scale}"
            results[f"A_{key}_z2_f1"] = float(np.mean(z2_f1s))
            results[f"A_{key}_topk_f1"] = float(np.mean(topk_f1s))
            results[f"A_{key}_peak_f1"] = float(np.mean(peak_f1s))
            results[f"A_{key}_uni_f1"] = float(np.mean(uni_f1s))

            z2_f1_all.extend(z2_f1s)
            uniform_f1_all.extend(uni_f1s)
            topk_f1_all.extend(topk_f1s)
            peak_f1_all.extend(peak_f1s)

    results["A_z2_f1_mean"] = float(np.mean(z2_f1_all)) if z2_f1_all else 0.0
    results["A_best_baseline_f1"] = max(
        float(np.mean(uniform_f1_all)) if uniform_f1_all else 0.0,
        float(np.mean(topk_f1_all)) if topk_f1_all else 0.0,
        float(np.mean(peak_f1_all)) if peak_f1_all else 0.0,
    )

    # ── Test B: Lambda Sweep ──────────────────────────────────────────
    best_lam_f1 = 0.0
    for lam_val in lambda_values:
        f1_list: List[float] = []
        for trial in range(n_trials):
            traj, gt_events = _generate_event_sparse(d, T, default_K, default_r, rng)
            traj = traj + rng.standard_normal(traj.shape) * 0.1
            scores = sharp_event_score(traj)
            y_star = solve_relaxed_selector(scores, default_K, default_r, lam_val, solver="osqp")
            I_star = hard_projection(y_star, default_K, default_r)
            f1_list.append(match_f1(I_star, gt_events, tolerance))
        mean_f1 = float(np.mean(f1_list))
        results[f"B_lam{lam_val}_f1"] = mean_f1
        best_lam_f1 = max(best_lam_f1, mean_f1)

    results["B_best_lambda_f1"] = best_lam_f1

    # ── Test C: Refractory Separation Effect ──────────────────────────
    for r_val in r_values:
        for spacing in ["wide", "narrow"]:
            min_gap = max(r_val + 2, 5) if spacing == "wide" else max(r_val, 1)
            f1_list = []
            for trial in range(n_trials):
                traj, gt_events = _generate_event_sparse(
                    d, T, default_K, r_val, rng, min_gap=min_gap,
                )
                scores = sharp_event_score(traj)
                y_star = solve_relaxed_selector(scores, default_K, r_val, default_lam, solver="osqp")
                I_star = hard_projection(y_star, default_K, r_val)
                f1_list.append(match_f1(I_star, gt_events, tolerance))
            results[f"C_r{r_val}_{spacing}_f1"] = float(np.mean(f1_list))

    save_experiment_npz("EZ2-03", seed, {"last_traj": traj, "gt_events": np.array(gt_events), "y_star": y_star}, cfg.output_dir)

    return results


# ── Entry Point ────────────────────────────────────────────────────────────

def run_experiment(config: Any = None) -> Dict[str, Any]:
    """Run EZ2-03 across all seeds and return aggregated report."""
    if config is None:
        config = load_emp_config("EZ2-03")

    report = run_multi_seed(
        experiment_fn=run_single_seed,
        config=config,
        seeds=config.training.seeds,
        experiment_id="EZ2-03",
        output_dir=config.output_dir,
    )

    agg = report["aggregated"]
    z2_mean = agg.get("A_z2_f1_mean", {}).get("mean", 0.0)
    baseline_mean = agg.get("A_best_baseline_f1", {}).get("mean", 0.0)
    margin = 0.02
    if hasattr(config, "acceptance_gates"):
        margin = getattr(config.acceptance_gates, "f1_margin", margin)

    passed = z2_mean >= baseline_mean - margin

    report["experiment_name"] = "Event Sparse Recovery"
    report["pass_criterion"] = f"z2_f1 >= best_baseline_f1 - {margin}"
    report["passed"] = passed

    log.info("[EZ2-03] z2_f1=%.4f  baseline_f1=%.4f  passed=%s",
             z2_mean, baseline_mean, passed)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if str(Path(__file__).resolve().parent.parent.parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
    result = run_experiment()
    print(f"\nEZ2-03 PASSED: {result['passed']}")
