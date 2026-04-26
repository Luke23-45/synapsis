"""
Topology feature analysis and visualization.

This module turns topology-token inspection into a reproducible artifact
generation step with explicit validation, summary statistics, and a stable
embedding export for later review.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def _validate_results(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    tokens = np.asarray(data.get("topology_tokens"))
    phases = np.asarray(data.get("phase_labels"))
    if tokens.size == 0 or phases.size == 0:
        raise ValueError("Topology analysis received no samples.")
    if tokens.ndim != 2:
        raise ValueError(f"Expected topology_tokens to be rank-2, got shape {tokens.shape}")
    if phases.ndim != 1:
        phases = phases.reshape(-1)
    if tokens.shape[0] != phases.shape[0]:
        raise ValueError(
            f"Mismatch between topology tokens ({tokens.shape[0]}) and phase labels ({phases.shape[0]})."
        )
    return tokens.astype(np.float32, copy=False), phases.astype(np.int32, copy=False)


@torch.no_grad()
def collect_topology_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[str, Any]:
    """Collect topology-token features and sample metadata."""
    model.eval()
    all_tokens = []
    all_phases = []
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
        all_tokens.append(output.topology_token.detach().cpu().numpy())
        all_phases.append(batch["phase_label"].detach().cpu().numpy())
        num_batches += 1

    data = {
        "topology_tokens": np.concatenate(all_tokens, axis=0) if all_tokens else np.empty((0, 0), dtype=np.float32),
        "phase_labels": np.concatenate(all_phases, axis=0) if all_phases else np.empty((0,), dtype=np.int32),
        "num_batches": num_batches,
    }
    _validate_results(data)
    return data


def _compute_embedding(
    tokens: np.ndarray,
    max_samples: int,
    perplexity: int,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    sample_count = min(max_samples, tokens.shape[0])
    if sample_count < 2:
        raise ValueError("Need at least two topology samples for embedding.")

    if sample_count < tokens.shape[0]:
        rng = np.random.default_rng(random_seed)
        selected = np.sort(rng.choice(tokens.shape[0], size=sample_count, replace=False))
    else:
        selected = np.arange(tokens.shape[0])

    sampled_tokens = tokens[selected]
    try:
        from sklearn.manifold import TSNE

        effective_perplexity = max(2, min(perplexity, sampled_tokens.shape[0] - 1))
        embedding = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            random_state=random_seed,
            init="pca",
            learning_rate="auto",
        ).fit_transform(sampled_tokens)
        return embedding.astype(np.float32), selected, "tsne"
    except Exception as exc:
        log.warning("Falling back to PCA for topology embedding: %s", exc)

    centered = sampled_tokens - sampled_tokens.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    embedding = centered @ vh[:2].T
    return embedding.astype(np.float32), selected, "pca"


def plot_topology_tsne(
    data: Dict[str, Any],
    save_dir: Path,
    title: str = "Topology Token Embedding",
    max_samples: int = 2000,
    perplexity: int = 30,
    random_seed: int = 42,
) -> None:
    """Create a 2D embedding plot and export the embedding table."""
    tokens, phases = _validate_results(data)
    save_dir.mkdir(parents=True, exist_ok=True)

    embedding, selected, method = _compute_embedding(
        tokens=tokens,
        max_samples=max_samples,
        perplexity=perplexity,
        random_seed=random_seed,
    )
    selected_phases = phases[selected]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 7))
    unique_phases = np.unique(selected_phases)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_phases)))

    for phase, color in zip(unique_phases, colors):
        mask = selected_phases == phase
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[color],
            label=f"Phase {phase}",
            alpha=0.7,
            s=18,
            edgecolors="none",
        )

    ax.set_title(f"{title} ({method.upper()})", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{method.upper()} 1", fontsize=11)
    ax.set_ylabel(f"{method.upper()} 2", fontsize=11)
    ax.legend(fontsize=10, markerscale=2, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_dir / "topology_embedding.png", dpi=220, bbox_inches="tight")
    fig.savefig(save_dir / "topology_embedding.pdf", bbox_inches="tight")
    plt.close(fig)

    with open(save_dir / "topology_embedding.csv", "w", encoding="utf-8") as handle:
        handle.write("sample_index,phase_label,x,y\n")
        for sample_index, phase, coords in zip(selected, selected_phases, embedding):
            handle.write(f"{int(sample_index)},{int(phase)},{coords[0]:.8f},{coords[1]:.8f}\n")

    log.info("Saved topology embedding plot and table to %s", save_dir)


def compute_topology_statistics(data: Dict[str, Any]) -> Dict[str, float]:
    """Compute publication-facing topology summary metrics."""
    tokens, phases = _validate_results(data)

    norms = np.linalg.norm(tokens, axis=1)
    feature_std = tokens.std(axis=0).mean()
    feature_mean_abs = np.abs(tokens).mean()
    unique_phases = np.unique(phases)

    fisher_ratio = 0.0
    min_phase_count = float("inf")
    if unique_phases.shape[0] > 1:
        class_means = []
        intra_vars = []
        global_mean = tokens.mean(axis=0)
        between_acc = 0.0
        within_acc = 0.0
        for phase in unique_phases:
            phase_tokens = tokens[phases == phase]
            min_phase_count = min(min_phase_count, phase_tokens.shape[0])
            phase_mean = phase_tokens.mean(axis=0)
            class_means.append(phase_mean)
            phase_var = phase_tokens.var(axis=0).mean()
            intra_vars.append(phase_var)
            between_acc += phase_tokens.shape[0] * float(np.mean((phase_mean - global_mean) ** 2))
            within_acc += phase_tokens.shape[0] * float(phase_var)
        fisher_ratio = between_acc / max(within_acc, 1e-8)
        inter_var = np.var(np.stack(class_means, axis=0), axis=0).mean()
        intra_var = float(np.mean(intra_vars))
        phase_separability = float(inter_var / max(intra_var, 1e-8))
    else:
        min_phase_count = float(tokens.shape[0])
        phase_separability = 0.0

    return {
        "num_samples": float(tokens.shape[0]),
        "num_phases": float(unique_phases.shape[0]),
        "min_phase_count": float(min_phase_count),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "feature_variability": float(feature_std),
        "feature_mean_abs": float(feature_mean_abs),
        "phase_separability": phase_separability,
        "fisher_ratio": float(fisher_ratio),
    }


def save_topology_analysis(data: Dict[str, Any], save_dir: Path) -> Dict[str, float]:
    """Persist topology statistics as JSON for later review."""
    stats = compute_topology_statistics(data)
    _, phases = _validate_results(data)
    payload = {
        "stats": stats,
        "num_batches": int(data["num_batches"]),
        "phase_histogram": {
            str(phase): int(count)
            for phase, count in zip(*np.unique(phases, return_counts=True))
        },
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "topology_analysis.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return stats
