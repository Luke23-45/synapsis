"""
Domain-specific plotting for SYNAPSE experiment verification.

Each plot function produces publication-quality figures for a specific
formal verification result, following the GibbsQ plotting pattern:
- Theme-aware (publication / dark)
- Multi-format export via chart_exporter
- Colorblind-safe palettes
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments.common.theme import apply_theme, get_current_theme, get_plot_colors, THEMES
from experiments.common.chart_exporter import save_chart

log = logging.getLogger(__name__)

__all__ = [
    "plot_event_scores",
    "plot_anchor_selection",
    "plot_changepoint_identification",
    "plot_reconstruction_error",
    "plot_topological_stability",
    "plot_cardinality_vs_K",
    "plot_collision_rates",
    "plot_experiment_summary",
]


def _setup(theme: Optional[str] = None) -> str:
    """Apply theme and return theme name."""
    theme = theme or get_current_theme() or "publication"
    apply_theme(theme)
    return theme


# -----------------------------------------------------------------------
# EXP-01/06: Event score visualisation
# -----------------------------------------------------------------------

def plot_event_scores(
    trajectory: np.ndarray,
    scores: np.ndarray,
    anchor_indices: Optional[List[int]] = None,
    tau: Optional[float] = None,
    title: str = "Event Score Sequence",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[List[str]] = None,
    theme: Optional[str] = None,
) -> plt.Figure:
    """
    Plot event scores e_t along with optional threshold and anchor markers.

    Parameters
    ----------
    trajectory : ndarray (T, d)
    scores : ndarray (T,)
    anchor_indices : list of int, optional
    tau : float, optional
    """
    theme = _setup(theme)
    colors = get_plot_colors(theme)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                              gridspec_kw={"height_ratios": [1, 2]})

    T = len(scores)
    t = np.arange(T)

    # Top panel: trajectory norm
    ax_traj = axes[0]
    traj_norms = np.linalg.norm(trajectory, axis=1)
    ax_traj.plot(t, traj_norms, color=colors["primary"], alpha=0.8, linewidth=1.0)
    ax_traj.set_ylabel(r"$\|x_t\|_2$")
    ax_traj.set_title(title)

    # Bottom panel: event scores
    ax_score = axes[1]
    ax_score.fill_between(t, 0, scores, alpha=0.3, color=colors["secondary"])
    ax_score.plot(t, scores, color=colors["secondary"], linewidth=1.0)

    if tau is not None:
        ax_score.axhline(tau, color=colors["error"], linestyle="--",
                         linewidth=1.5, alpha=0.8, label=rf"$\tau = {tau}$")

    if anchor_indices:
        ax_score.scatter(
            anchor_indices,
            [scores[i] for i in anchor_indices],
            color=colors["error"], s=50, zorder=5,
            marker="v", label=f"Anchors (m={len(anchor_indices)})",
        )
        for idx in anchor_indices:
            ax_traj.axvline(idx, color=colors["error"], alpha=0.3, linewidth=0.8)

    ax_score.set_xlabel("Time index $t$")
    ax_score.set_ylabel(r"Event score $e_t$")
    ax_score.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"], close_fig=False)

    return fig


# -----------------------------------------------------------------------
# EXP-02: Cardinality vs K
# -----------------------------------------------------------------------

def plot_cardinality_vs_K(
    K_values: List[int],
    max_m_observed: List[int],
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[List[str]] = None,
    theme: Optional[str] = None,
) -> plt.Figure:
    """Bar chart: maximum observed |I*| vs budget K."""
    theme = _setup(theme)
    colors = get_plot_colors(theme)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(K_values))
    palette = THEMES[theme].color_palette

    bars = ax.bar(x, max_m_observed, color=palette[1], alpha=0.85,
                  edgecolor=colors["contour"], label=r"max $|I^*|$ observed")
    ax.plot(x, K_values, "o--", color=palette[5], linewidth=2.0,
            markersize=8, label="Budget $K$")

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in K_values])
    ax.set_xlabel("Anchor Budget $K$")
    ax.set_ylabel("Anchor Count")
    ax.set_title(r"Bounded Cardinality: $|I^*| \leq K$")
    ax.legend()

    for bar, k, m in zip(bars, K_values, max_m_observed):
        color = palette[2] if m <= k else palette[5]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(m), ha="center", va="bottom", fontsize=9, color=color)

    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"], close_fig=False)

    return fig


# -----------------------------------------------------------------------
# EXP-03: Change-point identification
# -----------------------------------------------------------------------

def plot_changepoint_identification(
    scores: np.ndarray,
    true_changepoints: List[int],
    detected_anchors: List[int],
    tau: float,
    title: str = "Change-Point Identification",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[List[str]] = None,
    theme: Optional[str] = None,
) -> plt.Figure:
    """Visualise true change-points vs detected anchors."""
    theme = _setup(theme)
    colors = get_plot_colors(theme)
    palette = THEMES[theme].color_palette

    fig, ax = plt.subplots(figsize=(10, 4))

    T = len(scores)
    t = np.arange(T)

    ax.fill_between(t, 0, scores, alpha=0.2, color=palette[1])
    ax.plot(t, scores, color=palette[1], linewidth=1.0, label="Event scores")
    ax.axhline(tau, color=palette[5], linestyle="--", linewidth=1.5,
               alpha=0.7, label=rf"$\tau = {tau}$")

    # True change points
    for i, cp in enumerate(true_changepoints):
        ax.axvline(cp, color=palette[2], linewidth=2.0, alpha=0.5,
                   label="True $c_j$" if i == 0 else None)

    # Detected anchors
    ax.scatter(
        detected_anchors,
        [scores[i] for i in detected_anchors],
        color=palette[5], s=80, zorder=5, marker="^",
        edgecolors=colors["contour"], linewidths=0.5,
        label=f"Detected $I^*$ (m={len(detected_anchors)})",
    )

    # Mark matches / mismatches
    tp = set(true_changepoints) & set(detected_anchors)
    fp = set(detected_anchors) - set(true_changepoints)
    fn = set(true_changepoints) - set(detected_anchors)

    status_text = f"TP={len(tp)}  FP={len(fp)}  FN={len(fn)}"
    passed = len(fp) == 0 and len(fn) == 0
    status_color = palette[2] if passed else palette[5]

    ax.text(0.98, 0.95, status_text, transform=ax.transAxes,
            fontsize=10, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.9, edgecolor=status_color))

    ax.set_xlabel("Time index $t$")
    ax.set_ylabel(r"Event score $e_t$")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"], close_fig=False)

    return fig


# -----------------------------------------------------------------------
# EXP-04: Reconstruction error
# -----------------------------------------------------------------------

def plot_reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
    anchor_indices: List[int],
    title: str = "Trajectory Reconstruction",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[List[str]] = None,
    theme: Optional[str] = None,
) -> plt.Figure:
    """Plot original vs reconstructed trajectory with error."""
    theme = _setup(theme)
    colors = get_plot_colors(theme)
    palette = THEMES[theme].color_palette

    T = original.shape[0]
    t = np.arange(T)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    # Top: original vs reconstructed (1st dimension)
    ax_top = axes[0]
    ax_top.plot(t, original[:, 0], color=palette[0], linewidth=1.5,
                alpha=0.8, label="Original $x_t$")
    ax_top.plot(t, reconstructed[:, 0], color=palette[5], linewidth=1.5,
                linestyle="--", alpha=0.8, label=r"Reconstructed $\hat{x}_t$")

    for idx in anchor_indices:
        ax_top.axvline(idx, color=palette[2], alpha=0.3, linewidth=0.8)

    ax_top.set_ylabel("$x_t[0]$ (first dimension)")
    ax_top.set_title(title)
    ax_top.legend(loc="upper right")

    # Bottom: pointwise error
    ax_bot = axes[1]
    pointwise_error = np.linalg.norm(original - reconstructed, axis=1)
    ax_bot.semilogy(t, pointwise_error + 1e-16, color=palette[5], linewidth=1.0)
    ax_bot.fill_between(t, 1e-16, pointwise_error + 1e-16,
                        alpha=0.3, color=palette[5])
    ax_bot.set_xlabel("Time index $t$")
    ax_bot.set_ylabel(r"$\|\hat{x}_t - x_t\|_2$")

    max_err = np.max(pointwise_error)
    ax_bot.text(0.98, 0.85, f"max error = {max_err:.2e}",
                transform=ax_bot.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"], close_fig=False)

    return fig


# -----------------------------------------------------------------------
# EXP-05: Topological stability
# -----------------------------------------------------------------------

def plot_topological_stability(
    epsilon_values: List[float],
    hausdorff_ratios: List[float],
    bottleneck_ratios: List[float],
    title: str = "Topological Stability Bounds",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[List[str]] = None,
    theme: Optional[str] = None,
) -> plt.Figure:
    """Plot tightness ratios d_H/ε and d_B/(2ε) vs ε."""
    theme = _setup(theme)
    palette = THEMES[theme].color_palette

    fig, ax = plt.subplots(figsize=(8, 5))

    eps = np.array(epsilon_values)
    h_ratios = np.array(hausdorff_ratios)
    b_ratios = np.array(bottleneck_ratios)

    ax.plot(eps, h_ratios, "o-", color=palette[0], linewidth=2.0,
            markersize=8, label=r"$d_H / \varepsilon$")
    ax.plot(eps, b_ratios, "s-", color=palette[4], linewidth=2.0,
            markersize=8, label=r"$d_B / (2\varepsilon)$")

    # Bound line at 1.0
    ax.axhline(1.0, color=palette[5], linestyle="--", linewidth=1.5,
               alpha=0.7, label="Theoretical bound")

    ax.fill_between(eps, 0, 1.0, alpha=0.1, color=palette[2])
    ax.text(eps[len(eps)//2], 0.5, "VALID REGION", ha="center",
            fontsize=10, color=palette[2], alpha=0.6)

    ax.set_xscale("log")
    ax.set_xlabel(r"Perturbation $\varepsilon$")
    ax.set_ylabel("Tightness ratio")
    ax.set_title(title)
    ax.set_ylim(0, 1.5)
    ax.legend()

    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"], close_fig=False)

    return fig


# -----------------------------------------------------------------------
# EXP-07: Collision rates
# -----------------------------------------------------------------------

def plot_collision_rates(
    K_values: List[int],
    collision_rates: List[float],
    title: str = "Information Loss: Collision Rate vs $K$",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[List[str]] = None,
    theme: Optional[str] = None,
) -> plt.Figure:
    """Plot collision rate as a function of anchor budget K."""
    theme = _setup(theme)
    palette = THEMES[theme].color_palette

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(K_values, collision_rates, "o-", color=palette[4],
            linewidth=2.0, markersize=10)
    ax.fill_between(K_values, 0, collision_rates, alpha=0.15, color=palette[4])

    ax.set_xlabel("Anchor Budget $K$")
    ax.set_ylabel("Pairwise Collision Rate")
    ax.set_title(title)

    # Annotate monotonicity
    if len(collision_rates) >= 2:
        is_monotone = all(
            collision_rates[i] >= collision_rates[i+1]
            for i in range(len(collision_rates) - 1)
        )
        label = "Monotonic PASS" if is_monotone else "Non-monotonic FAIL"
        color = palette[2] if is_monotone else palette[5]
        ax.text(0.98, 0.95, label, transform=ax.transAxes,
                ha="right", va="top", fontsize=11, color=color,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"], close_fig=False)

    return fig


# -----------------------------------------------------------------------
# Master summary
# -----------------------------------------------------------------------

def plot_experiment_summary(
    experiment_names: List[str],
    statuses: List[str],
    durations: List[float],
    case_counts: List[Tuple[int, int]],
    title: str = "SYNAPSE Verification Suite — Summary",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[List[str]] = None,
    theme: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart summarising all experiments.

    Parameters
    ----------
    experiment_names : list of str
    statuses : list of str ('PASS', 'FAIL', 'ERROR')
    durations : list of float (seconds)
    case_counts : list of (passed, total) tuples
    """
    theme = _setup(theme)
    palette = THEMES[theme].color_palette
    colors_map = {"PASS": palette[2], "FAIL": palette[5], "ERROR": palette[0]}

    n = len(experiment_names)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, n * 0.6)),
                              gridspec_kw={"width_ratios": [3, 1]})

    # Left panel: pass rates
    ax_left = axes[0]
    y = np.arange(n)
    pass_rates = [p / max(t, 1) for p, t in case_counts]
    bar_colors = [colors_map.get(s, palette[7]) for s in statuses]

    bars = ax_left.barh(y, pass_rates, color=bar_colors, alpha=0.85,
                         edgecolor="white", height=0.6)

    for i, (bar, (p, t), status) in enumerate(zip(bars, case_counts, statuses)):
        icon = "PASS" if status == "PASS" else "FAIL"
        ax_left.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{icon} {p}/{t}", va="center", fontsize=9)

    ax_left.set_yticks(y)
    ax_left.set_yticklabels(experiment_names)
    ax_left.set_xlim(0, 1.15)
    ax_left.set_xlabel("Pass Rate")
    ax_left.set_title(title)
    ax_left.invert_yaxis()

    # Right panel: durations
    ax_right = axes[1]
    ax_right.barh(y, durations, color=palette[1], alpha=0.7, height=0.6)
    for i, d in enumerate(durations):
        ax_right.text(d + 0.01, i, f"{d:.1f}s", va="center", fontsize=8)
    ax_right.set_xlabel("Duration (s)")
    ax_right.set_yticks([])
    ax_right.invert_yaxis()

    plt.tight_layout()

    if save_path:
        save_chart(fig, Path(save_path), formats or ["png", "pdf"], close_fig=False)

    return fig
