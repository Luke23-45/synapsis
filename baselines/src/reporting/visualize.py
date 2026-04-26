"""
Publication-Quality Visualization — M1 Experiment
==================================================

Generates all comparison plots specified in the PLAN.md:
    1. Learning curves: Train/val MSE over epochs for all conditions
    2. Grouped bar chart: Final test MSE for A1, A2, B with error bars
    3. Scatter plot: Per-episode MSE (A1 vs B, A2 vs B)
    4. Stratified analysis: MSE by episode length quartile
    5. Memory budget sweep: MSE vs K (anchor count)
    6. Phase transition analysis: MSE near boundaries vs away
    7. Rollout error accumulation: MSE curve over rollout steps
    8. Ablation comparison: B-Full vs B-Anchors vs B-Topo
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    log.warning("matplotlib not available — visualization will be skipped")


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

CONDITION_COLORS = {
    "A1_recent": "#7f8c8d",
    "A2_uniform": "#3498db",
    "B_synapse": "#e74c3c",
    "B_anchors": "#e67e22",
    "B_topo": "#9b59b6",
}

CONDITION_LABELS = {
    "A1_recent": "A1: Recent Window",
    "A2_uniform": "A2: Uniform Subsampling",
    "B_synapse": "B: SYNAPSE Full",
    "B_anchors": "B-Anchors",
    "B_topo": "B-Topo",
}

FIG_DPI = 300
FIG_WIDTH = 8
FIG_HEIGHT = 5


def _setup_axes(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent styling to axes."""
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_learning_curves(
    train_losses: Dict[str, List[float]],
    val_losses: Dict[str, List[float]],
    output_path: Path,
) -> Optional[Path]:
    """Plot train/val MSE over epochs for all conditions."""
    if not HAS_MATPLOTLIB:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH * 2, FIG_HEIGHT))

    for cond_key, losses in train_losses.items():
        color = CONDITION_COLORS.get(cond_key, "#333333")
        label = CONDITION_LABELS.get(cond_key, cond_key)
        ax1.plot(losses, color=color, label=label, alpha=0.8)

    _setup_axes(ax1, "Training MSE", "Epoch", "MSE")
    ax1.legend(fontsize=8)

    for cond_key, losses in val_losses.items():
        color = CONDITION_COLORS.get(cond_key, "#333333")
        label = CONDITION_LABELS.get(cond_key, cond_key)
        ax2.plot(losses, color=color, label=label, alpha=0.8)

    _setup_axes(ax2, "Validation MSE", "Epoch", "MSE")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved learning curves to %s", output_path)
    return output_path


def plot_mse_comparison(
    condition_means: Dict[str, float],
    condition_stds: Dict[str, float],
    output_path: Path,
) -> Optional[Path]:
    """Grouped bar chart: Final test MSE for conditions with error bars."""
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    conditions = list(condition_means.keys())
    means = [condition_means[c] for c in conditions]
    stds = [condition_stds[c] for c in conditions]
    colors = [CONDITION_COLORS.get(c, "#333333") for c in conditions]
    labels = [CONDITION_LABELS.get(c, c) for c in conditions]

    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8,
                  capsize=5, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    _setup_axes(ax, "Action MSE Comparison", "Condition", "MSE")

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.001,
            f"{mean:.4f}",
            ha="center", va="bottom", fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved MSE comparison to %s", output_path)
    return output_path


def plot_per_episode_scatter(
    mse_a: np.ndarray,
    mse_b: np.ndarray,
    condition_a: str,
    condition_b: str,
    output_path: Path,
) -> Optional[Path]:
    """Scatter plot: Per-episode MSE for two conditions."""
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(FIG_HEIGHT, FIG_HEIGHT))

    n = min(len(mse_a), len(mse_b))
    ax.scatter(mse_a[:n], mse_b[:n], alpha=0.5, s=20, color="#3498db")

    # Diagonal reference line
    max_val = max(mse_a[:n].max(), mse_b[:n].max()) if n > 0 else 1.0
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="Equal")

    label_a = CONDITION_LABELS.get(condition_a, condition_a)
    label_b = CONDITION_LABELS.get(condition_b, condition_b)
    _setup_axes(ax, f"Per-Episode MSE: {label_a} vs {label_b}",
                label_a, label_b)
    ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved scatter plot to %s", output_path)
    return output_path


def plot_stratified_analysis(
    strata: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = "MSE by Episode Length Quartile",
) -> Optional[Path]:
    """Bar chart of MSE stratified by episode length or phase density."""
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    labels = list(strata.keys())
    means = [strata[k]["mean_mse"] for k in labels]
    stds = [strata[k]["std_mse"] for k in labels]

    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color="#3498db", alpha=0.8,
           capsize=5, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    _setup_axes(ax, title, "Stratum", "MSE")

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved stratified analysis to %s", output_path)
    return output_path


def plot_rollout_error_accumulation(
    rollout_curves: Dict[str, Dict],
    output_path: Path,
) -> Optional[Path]:
    """MSE curve over rollout steps for all conditions."""
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    for cond_key, agg in rollout_curves.items():
        color = CONDITION_COLORS.get(cond_key, "#333333")
        label = CONDITION_LABELS.get(cond_key, cond_key)
        mean_curve = np.array(agg["mean_mse_per_step"])
        std_curve = np.array(agg["std_mse_per_step"])
        steps = np.arange(len(mean_curve))

        ax.plot(steps, mean_curve, color=color, label=label, linewidth=2)
        ax.fill_between(
            steps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.15,
        )

    _setup_axes(ax, "Rollout Error Accumulation", "Rollout Step", "MSE")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved rollout error accumulation to %s", output_path)
    return output_path


def plot_ablation_comparison(
    full_mse: float,
    full_std: float,
    anchors_mse: float,
    anchors_std: float,
    topo_mse: float,
    topo_std: float,
    output_path: Path,
) -> Optional[Path]:
    """Bar chart: B-Full vs B-Anchors vs B-Topo."""
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.7, FIG_HEIGHT))

    labels = ["B-Full", "B-Anchors", "B-Topo"]
    means = [full_mse, anchors_mse, topo_mse]
    stds = [full_std, anchors_std, topo_std]
    colors = [CONDITION_COLORS["B_synapse"],
              CONDITION_COLORS["B_anchors"],
              CONDITION_COLORS["B_topo"]]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8,
                  capsize=5, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    _setup_axes(ax, "Ablation Comparison", "Condition", "MSE")

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{mean:.4f}",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved ablation comparison to %s", output_path)
    return output_path
