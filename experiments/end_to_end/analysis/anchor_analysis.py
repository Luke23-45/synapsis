"""
Anchor placement analysis and visualization for the end-to-end stack.

The goal of this module is not just to draw a heatmap. It also validates
the collected selector activations, computes stable summary statistics,
and writes explicit analysis artifacts that can be audited later.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def _as_numpy_sequence(values: List[np.ndarray]) -> List[np.ndarray]:
    return [np.asarray(value, dtype=np.float32).reshape(-1) for value in values]


def _validate_results(results: Dict[str, Any]) -> None:
    required = ("y_star_all", "history_lengths", "timesteps", "phase_labels")
    missing = [key for key in required if key not in results]
    if missing:
        raise ValueError(f"Anchor analysis results missing keys: {missing}")
    if not results["y_star_all"]:
        raise ValueError("Anchor analysis received no samples.")


def _sample_indices(count: int, target: int) -> np.ndarray:
    if target >= count:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, num=target, dtype=int))


@torch.no_grad()
def collect_anchor_placements(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[str, Any]:
    """Collect per-sample selector activations and metadata."""
    model.eval()
    y_star_all: List[np.ndarray] = []
    history_lengths: List[int] = []
    timesteps: List[int] = []
    phase_labels: List[int] = []

    num_batches = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if not batch:
            continue

        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        output = model.forward_train(batch)
        num_batches += 1

        for sample_idx in range(output.y_star.shape[0]):
            history_length = int(batch["history_lengths"][sample_idx].item())
            y_star_all.append(output.y_star[sample_idx, :history_length].detach().cpu().numpy())
            history_lengths.append(history_length)
            timesteps.append(int(batch["timestep"][sample_idx].item()))
            phase_labels.append(int(batch["phase_label"][sample_idx].item()))

    results = {
        "y_star_all": _as_numpy_sequence(y_star_all),
        "history_lengths": np.asarray(history_lengths, dtype=np.int32),
        "timesteps": np.asarray(timesteps, dtype=np.int32),
        "phase_labels": np.asarray(phase_labels, dtype=np.int32),
        "num_batches": num_batches,
        "num_samples": len(y_star_all),
    }
    _validate_results(results)
    return results


def plot_anchor_heatmap(
    results: Dict[str, Any],
    save_dir: Path,
    num_samples: int = 20,
    title: str = "Anchor Placement Heatmap",
) -> None:
    """Plot selector activations with padding-aware masking and length ordering."""
    _validate_results(results)
    save_dir.mkdir(parents=True, exist_ok=True)

    sequences = results["y_star_all"]
    lengths = np.asarray(results["history_lengths"])
    order = np.argsort(lengths)[::-1]
    chosen = order[_sample_indices(len(order), num_samples)]

    max_len = int(lengths[chosen].max())
    matrix = np.full((len(chosen), max_len), np.nan, dtype=np.float32)
    for row_idx, sample_idx in enumerate(chosen):
        seq = np.asarray(sequences[sample_idx], dtype=np.float32)
        matrix[row_idx, : seq.shape[0]] = seq

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_heatmap, ax_lengths) = plt.subplots(
        2,
        1,
        figsize=(14, max(6, len(chosen) * 0.32)),
        sharex=False,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    masked = np.ma.masked_invalid(matrix)
    image = ax_heatmap.imshow(masked, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax_heatmap.set_title(title, fontsize=14, fontweight="bold")
    ax_heatmap.set_ylabel("Sample (sorted by history length)", fontsize=11)
    ax_heatmap.set_xlabel("History step", fontsize=11)
    fig.colorbar(image, ax=ax_heatmap, label="selector activation", shrink=0.85)

    ax_lengths.bar(np.arange(len(chosen)), lengths[chosen], color="#5C6BC0", alpha=0.85)
    ax_lengths.set_ylabel("Length", fontsize=10)
    ax_lengths.set_xlabel("Displayed sample", fontsize=10)
    ax_lengths.spines["top"].set_visible(False)
    ax_lengths.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_dir / "anchor_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(save_dir / "anchor_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("Saved anchor heatmap to %s", save_dir)


def compute_anchor_statistics(
    results: Dict[str, Any],
    K: int = 10,
    activation_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute robust summary statistics for selector behavior."""
    _validate_results(results)

    sequences = [np.asarray(seq, dtype=np.float32) for seq in results["y_star_all"]]
    lengths = np.asarray(results["history_lengths"], dtype=np.float32)

    active_counts = np.asarray([np.sum(seq > activation_threshold) for seq in sequences], dtype=np.float32)
    max_activations = np.asarray([float(np.max(seq)) if seq.size else 0.0 for seq in sequences], dtype=np.float32)
    mean_activations = np.asarray([seq.mean() if seq.size else 0.0 for seq in sequences], dtype=np.float32)
    sparsity_ratios = np.asarray([np.mean(seq < 0.1) if seq.size else 1.0 for seq in sequences], dtype=np.float32)
    activation_mass = np.asarray([seq.sum() for seq in sequences], dtype=np.float32)

    temporal_centers = []
    entropies = []
    for seq in sequences:
        if seq.size == 0 or np.allclose(seq.sum(), 0.0):
            temporal_centers.append(np.nan)
            entropies.append(np.nan)
            continue
        weights = np.clip(seq, a_min=0.0, a_max=None)
        weights = weights / max(weights.sum(), 1e-8)
        positions = np.linspace(0.0, 1.0, num=seq.size, dtype=np.float32)
        temporal_centers.append(float(np.sum(weights * positions)))
        entropies.append(float(-np.sum(weights * np.log(np.clip(weights, 1e-8, None)))))

    temporal_centers_arr = np.asarray(temporal_centers, dtype=np.float32)
    entropies_arr = np.asarray(entropies, dtype=np.float32)

    return {
        "num_samples": float(len(sequences)),
        "mean_history_length": float(lengths.mean()),
        "std_history_length": float(lengths.std()),
        "mean_active": float(active_counts.mean()),
        "median_active": float(np.median(active_counts)),
        "utilization": float(active_counts.mean()) / max(K, 1),
        "mean_max_activation": float(max_activations.mean()),
        "mean_activation": float(mean_activations.mean()),
        "mean_activation_mass": float(activation_mass.mean()),
        "sparsity": float(sparsity_ratios.mean()),
        "mean_temporal_center": float(np.nanmean(temporal_centers_arr)),
        "std_temporal_center": float(np.nanstd(temporal_centers_arr)),
        "mean_activation_entropy": float(np.nanmean(entropies_arr)),
    }


def save_anchor_analysis(
    results: Dict[str, Any],
    save_dir: Path,
    K: int = 10,
    activation_threshold: float = 0.5,
) -> Dict[str, float]:
    """Persist anchor statistics for auditability."""
    stats = compute_anchor_statistics(results, K=K, activation_threshold=activation_threshold)
    payload = {
        "stats": stats,
        "num_batches": int(results["num_batches"]),
        "num_samples": int(results["num_samples"]),
        "phase_histogram": {
            str(phase): int(count)
            for phase, count in zip(*np.unique(results["phase_labels"], return_counts=True))
        },
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "anchor_analysis.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return stats
