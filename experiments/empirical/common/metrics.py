"""
experiments/empirical/common/metrics.py

Statistical testing and metric computation for multi-seed experiments.
Uses scipy for hypothesis tests and sklearn for classification metrics.

Phase 2 Shared Infrastructure — see docs/implementation/phase2_empirical_validation/07_shared_infrastructure.md §3
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


def ci95(values: List[float]) -> Tuple[float, float]:
    """95% confidence interval assuming normal distribution."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return (float(arr[0]), float(arr[0])) if n == 1 else (0.0, 0.0)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(n))
    margin = 1.96 * se
    return (mean - margin, mean + margin)


def paired_ttest(a: List[float], b: List[float]) -> Dict[str, float]:
    """
    Paired t-test: H0: mean(a) == mean(b).
    Returns t-statistic, p-value, and Cohen's d.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) < 2 or len(b_arr) < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "cohens_d": 0.0}
    t_stat, p_value = stats.ttest_rel(a_arr, b_arr)
    diff = a_arr - b_arr
    cohens_d = float(diff.mean() / (diff.std(ddof=1) + 1e-10))
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": cohens_d,
    }


def wilcoxon_test(a: List[float], b: List[float]) -> Dict[str, float]:
    """Non-parametric alternative to paired t-test."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    diff = a_arr - b_arr
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0}
    try:
        stat, p_value = stats.wilcoxon(diff)
        return {"statistic": float(stat), "p_value": float(p_value)}
    except ValueError:
        return {"statistic": 0.0, "p_value": 1.0}


def bonferroni_correct(p_values: List[float], alpha: float = 0.05) -> List[Dict]:
    """Apply Bonferroni correction to multiple p-values."""
    k = len(p_values)
    if k == 0:
        return []
    corrected_alpha = alpha / k
    return [
        {
            "raw_p": p,
            "corrected_alpha": corrected_alpha,
            "significant": p < corrected_alpha,
        }
        for p in p_values
    ]


def aggregate_seeds(
    per_seed_results: List[Dict[str, float]],
    key: str,
) -> Dict[str, float]:
    """Aggregate a metric across seeds: mean, std, CI95, min, max."""
    values = [r[key] for r in per_seed_results]
    lo, hi = ci95(values)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_seeds": len(values),
        "raw_values": values,
    }


def match_f1(
    detected: List[int],
    ground_truth: List[int],
    tolerance: int,
) -> float:
    """F1 score for event detection with tolerance window.

    Uses ``scipy.optimize.linear_sum_assignment`` for globally optimal
    matching instead of the O(|gt| × |det|) greedy approach.

    A ground-truth event at position g is considered "hit" if matched
    to a detected event d with |d - g| ≤ tolerance.
    """
    if not ground_truth:
        return 1.0 if not detected else 0.0
    if not detected:
        return 0.0

    from scipy.optimize import linear_sum_assignment

    gt_arr = np.asarray(ground_truth, dtype=np.float64)
    det_arr = np.asarray(detected, dtype=np.float64)

    # Vectorised cost matrix: |det[i] - gt[j]|
    cost = np.abs(det_arr[:, None] - gt_arr[None, :])  # (n_det, n_gt)

    # Optimal assignment minimising total distance
    row_ind, col_ind = linear_sum_assignment(cost)

    # Count hits within tolerance
    hits = int(np.sum(cost[row_ind, col_ind] <= tolerance))

    precision = hits / len(detected)
    recall = hits / len(ground_truth)
    if precision + recall < 1e-10:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
