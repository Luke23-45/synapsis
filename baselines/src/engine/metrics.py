"""
Statistical Testing — M1 Experiment
====================================

Implements the statistical rigor protocol (§2.6 of PLAN.md):
    - Primary: Two-sided Welch's t-test (α=0.05)
    - Effect size: Cohen's d with bootstrapped 95% CI
    - Normality check: Shapiro-Wilk on paired differences
    - Non-parametric backup: Wilcoxon signed-rank test
    - Multiple comparison correction: Bonferroni for stratified analyses
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    log.warning("scipy not available — statistical tests will use fallback implementations")


# ---------------------------------------------------------------------------
# Core statistical tests
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Result of a statistical comparison between two conditions.

    Attributes
    ----------
    metric_name : str
        Name of the metric being compared.
    condition_a : str
    condition_b : str
    mean_a : float
    std_a : float
    mean_b : float
    std_b : float
    mean_diff : float
        mean_b - mean_a (positive = B better if lower is better for metric)
    cohens_d : float
        Effect size.
    d_ci_lower : float
        Lower bound of 95% CI for Cohen's d.
    d_ci_upper : float
        Upper bound of 95% CI for Cohen's d.
    t_statistic : float
        Welch's t-statistic.
    p_value : float
        Two-sided p-value from Welch's t-test.
    welch_significant : bool
        Whether p < α (default 0.05).
    shapiro_p : float
        p-value from Shapiro-Wilk normality test on paired differences.
    is_normal : bool
        Whether paired differences are approximately normal (p > 0.05).
    wilcoxon_p : float or None
        p-value from Wilcoxon signed-rank test (non-parametric backup).
    wilcoxon_significant : bool or None
    bonferroni_alpha : float
        Bonferroni-corrected α for this comparison.
    significant_after_correction : bool
    """

    metric_name: str
    condition_a: str
    condition_b: str
    mean_a: float
    std_a: float
    mean_b: float
    std_b: float
    mean_diff: float
    cohens_d: float
    d_ci_lower: float
    d_ci_upper: float
    t_statistic: float
    p_value: float
    welch_significant: bool
    shapiro_p: float
    is_normal: bool
    wilcoxon_p: Optional[float]
    wilcoxon_significant: Optional[bool]
    bonferroni_alpha: float
    significant_after_correction: bool


def cohens_d(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """Compute Cohen's d effect size.

    d = (mean_b - mean_a) / pooled_std

    Pooled standard deviation:
        s_p = sqrt(((n_a - 1) * s_a² + (n_b - 1) * s_b²) / (n_a + n_b - 2))
    """
    n_a, n_b = len(samples_a), len(samples_b)
    if n_a < 2 or n_b < 2:
        return 0.0

    mean_a, mean_b = np.mean(samples_a), np.mean(samples_b)
    var_a, var_b = np.var(samples_a, ddof=1), np.var(samples_b, ddof=1)

    pooled_std = np.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )

    if pooled_std < 1e-15:
        return 0.0

    return float((mean_b - mean_a) / pooled_std)


def bootstrap_ci_cohens_d(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrapped confidence interval for Cohen's d.

    Returns
    -------
    (lower, upper) : tuple of float
        Confidence interval bounds.
    """
    rng = np.random.default_rng(seed)
    n_a, n_b = len(samples_a), len(samples_b)

    if n_a < 2 or n_b < 2:
        return (0.0, 0.0)

    boot_ds = []
    for _ in range(n_bootstrap):
        idx_a = rng.choice(n_a, size=n_a, replace=True)
        idx_b = rng.choice(n_b, size=n_b, replace=True)
        d = cohens_d(samples_a[idx_a], samples_b[idx_b])
        boot_ds.append(d)

    boot_ds = np.array(boot_ds)
    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_ds, 100 * alpha / 2))
    upper = float(np.percentile(boot_ds, 100 * (1 - alpha / 2)))

    return lower, upper


def welch_t_test(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> Tuple[float, float]:
    """Two-sided Welch's t-test.

    Returns
    -------
    (t_statistic, p_value) : tuple of float
    """
    if HAS_SCIPY:
        t_stat, p_val = sp_stats.ttest_ind(
            samples_a, samples_b, equal_var=False
        )
        return float(t_stat), float(p_val)
    else:
        # Fallback: manual Welch's t-test
        n_a, n_b = len(samples_a), len(samples_b)
        mean_a, mean_b = np.mean(samples_a), np.mean(samples_b)
        var_a, var_b = np.var(samples_a, ddof=1), np.var(samples_b, ddof=1)

        se = np.sqrt(var_a / n_a + var_b / n_b)
        if se < 1e-15:
            return 0.0, 1.0

        t_stat = (mean_a - mean_b) / se

        # Welch-Satterthwaite degrees of freedom
        num = (var_a / n_a + var_b / n_b) ** 2
        denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        if denom < 1e-15:
            return float(t_stat), 1.0

        df = num / denom

        # Approximate p-value using normal distribution for large df
        from math import erfc, sqrt
        z = abs(t_stat)
        p_val = erfc(z / sqrt(2))

        return float(t_stat), float(p_val)


def shapiro_wilk_test(samples: np.ndarray) -> Tuple[float, float]:
    """Shapiro-Wilk normality test.

    Returns
    -------
    (statistic, p_value) : tuple of float
    """
    if HAS_SCIPY:
        stat, p_val = sp_stats.shapiro(samples)
        return float(stat), float(p_val)
    else:
        # Fallback: assume normality
        return 1.0, 1.0


def wilcoxon_test(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> Tuple[float, float]:
    """Wilcoxon signed-rank test (non-parametric backup).

    Returns
    -------
    (statistic, p_value) : tuple of float
    """
    if HAS_SCIPY:
        diff = samples_b - samples_a
        # Remove zeros (no change)
        diff = diff[diff != 0]
        if len(diff) < 2:
            return 0.0, 1.0
        stat, p_val = sp_stats.wilcoxon(diff)
        return float(stat), float(p_val)
    else:
        return 0.0, 1.0


def compare_conditions(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_name: str,
    condition_a: str,
    condition_b: str,
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    num_comparisons: int = 1,
) -> ComparisonResult:
    """Full statistical comparison between two conditions.

    Parameters
    ----------
    scores_a : np.ndarray, shape (N,)
        Per-episode metric scores for condition A.
    scores_b : np.ndarray, shape (N,)
        Per-episode metric scores for condition B.
    metric_name : str
    condition_a, condition_b : str
    alpha : float
        Significance level.
    n_bootstrap : int
        Bootstrap samples for Cohen's d CI.
    num_comparisons : int
        Number of comparisons for Bonferroni correction.

    Returns
    -------
    ComparisonResult
    """
    bonferroni_alpha = alpha / max(1, num_comparisons)

    mean_a, std_a = float(np.mean(scores_a)), float(np.std(scores_a, ddof=1))
    mean_b, std_b = float(np.mean(scores_b)), float(np.std(scores_b, ddof=1))
    mean_diff = mean_b - mean_a

    # Cohen's d
    d = cohens_d(scores_a, scores_b)
    d_ci_lower, d_ci_upper = bootstrap_ci_cohens_d(
        scores_a, scores_b, n_bootstrap=n_bootstrap
    )

    # Welch's t-test
    t_stat, p_val = welch_t_test(scores_a, scores_b)

    # Normality check on paired differences
    paired_diff = scores_b - scores_a
    _, shapiro_p = shapiro_wilk_test(paired_diff)
    is_normal = shapiro_p > 0.05

    # Wilcoxon signed-rank (non-parametric backup)
    wil_p, wil_sig = None, None
    if not is_normal and len(paired_diff) >= 2:
        wil_stat, wil_p = wilcoxon_test(scores_a, scores_b)
        wil_sig = wil_p < bonferroni_alpha if wil_p is not None else None

    welch_sig = p_val < alpha
    sig_after_correction = p_val < bonferroni_alpha

    return ComparisonResult(
        metric_name=metric_name,
        condition_a=condition_a,
        condition_b=condition_b,
        mean_a=mean_a,
        std_a=std_a,
        mean_b=mean_b,
        std_b=std_b,
        mean_diff=mean_diff,
        cohens_d=d,
        d_ci_lower=d_ci_lower,
        d_ci_upper=d_ci_upper,
        t_statistic=t_stat,
        p_value=p_val,
        welch_significant=welch_sig,
        shapiro_p=shapiro_p,
        is_normal=is_normal,
        wilcoxon_p=wil_p,
        wilcoxon_significant=wil_sig,
        bonferroni_alpha=bonferroni_alpha,
        significant_after_correction=sig_after_correction,
    )


# ---------------------------------------------------------------------------
# Stratified analysis
# ---------------------------------------------------------------------------

def stratify_by_episode_length(
    episode_lengths: np.ndarray,
    scores: np.ndarray,
    n_quartiles: int = 4,
) -> Dict[str, np.ndarray]:
    """Stratify scores by episode length quartiles.

    Parameters
    ----------
    episode_lengths : np.ndarray, shape (N,)
    scores : np.ndarray, shape (N,)
    n_quartiles : int

    Returns
    -------
    dict mapping quartile_label → scores_array
    """
    quartile_edges = np.percentile(
        episode_lengths, np.linspace(0, 100, n_quartiles + 1)
    )

    result = {}
    for q in range(n_quartiles):
        lower = quartile_edges[q]
        upper = quartile_edges[q + 1]
        if q == n_quartiles - 1:
            mask = (episode_lengths >= lower) & (episode_lengths <= upper)
        else:
            mask = (episode_lengths >= lower) & (episode_lengths < upper)

        label = f"Q{q+1}_({lower:.0f}-{upper:.0f})"
        result[label] = scores[mask]

    return result


def stratify_by_phase_density(
    phase_boundaries: np.ndarray,
    scores: np.ndarray,
    episode_lengths: np.ndarray,
    n_bins: int = 3,
) -> Dict[str, np.ndarray]:
    """Stratify scores by phase-transition density.

    Density = n_phase_boundaries / episode_length.

    Parameters
    ----------
    phase_boundaries : np.ndarray, shape (N,)
        Number of phase boundaries per episode.
    scores : np.ndarray, shape (N,)
    episode_lengths : np.ndarray, shape (N,)
    n_bins : int

    Returns
    -------
    dict mapping density_label → scores_array
    """
    density = phase_boundaries / np.maximum(episode_lengths, 1)
    bin_edges = np.percentile(density, np.linspace(0, 100, n_bins + 1))

    result = {}
    for b in range(n_bins):
        lower = bin_edges[b]
        upper = bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (density >= lower) & (density <= upper)
        else:
            mask = (density >= lower) & (density < upper)

        label = f"density_{b+1}_({lower:.3f}-{upper:.3f})"
        result[label] = scores[mask]

    return result
