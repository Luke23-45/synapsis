"""
SYNAPSE Feature Cache — Offline M() Caching Pipeline
=====================================================

CRITICAL DESIGN DECISION (Professor Feedback #2):
    Calling M() inside the model's forward pass requires CPU-bound numpy
    computation per sample per epoch. The GPU sits idle waiting for the
    CPU — a classic PyTorch anti-pattern.

    Solution: Pre-compute all SYNAPSE features before training begins.
    This module implements the offline caching pipeline that iterates
    through the full dataset, applies M() to each sample's proprio_history,
    and saves the resulting anchor clouds and topological summaries to disk.

Caching pipeline (§4.5 of PLAN.md):
    1. Iterate through the full dataset (train + val + test)
    2. For each sample, extract proprio_history
    3. Apply Z-score normalization (see normalization.py)
    4. Call M(proprio_norm, K, r, tau, weights, Q)
    5. Extract state.point_cloud → pad to K rows → save as synapse_anchors
    6. Extract state.persistence_diagrams → compute summary stats → save as synapse_topo
    7. Store all features in a single .pt file per split

Fallback:
    If offline caching is not feasible (e.g., dataset too large for disk),
    move M() into the Dataset.__getitem__ method so the DataLoader's
    num_workers can compute features in parallel via multiprocessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch

from synapse_core import M

from src.core.config import ExperimentConfig
from src.core.normalization import NormalizationStats

log = logging.getLogger(__name__)


@dataclass
class CachedSynapseFeatures:
    """Pre-computed SYNAPSE features for a single sample.

    Attributes
    ----------
    anchors : np.ndarray, shape (K, D_anchor)
        Padded anchor cloud. If M() produced m < K anchors, rows m..K-1
        are zero-filled. A separate mask tracks which rows are real.
    anchor_mask : np.ndarray, shape (K,)
        Boolean mask: True for real anchor rows, False for padding.
    topo : np.ndarray, shape (D_topo,)
        Topological summary vector (4 statistics × (Q+1) diagram degrees).
    """

    anchors: np.ndarray
    anchor_mask: np.ndarray
    topo: np.ndarray


def pad_anchor_cloud(cloud: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad anchor cloud to exactly K rows with zero-fill + boolean mask.

    Parameters
    ----------
    cloud : np.ndarray, shape (m, D) or (0, 0)
        The lifted anchor cloud from M(). May have m < K anchors.
    K : int
        Target number of rows (the anchor budget).

    Returns
    -------
    padded : np.ndarray, shape (K, D)
        Zero-padded anchor cloud.
    mask : np.ndarray, shape (K,)
        Boolean mask where True indicates a real anchor row.
    """
    if cloud.size == 0 or cloud.shape[0] == 0:
        # No anchors at all — all padding
        D = 1  # Placeholder dimension; will be overridden by caller
        # When cloud is (0, 0), we need to know the expected feature dim
        # This is handled by the caller providing the expected dim
        return np.zeros((K, 0), dtype=np.float32), np.zeros(K, dtype=bool)

    m, D = cloud.shape
    if m > K:
        # More anchors than budget — truncate (shouldn't happen if M() respects K)
        log.warning(
            "Anchor cloud has %d points but budget K=%d. Truncating.", m, K
        )
        return cloud[:K].astype(np.float32), np.ones(K, dtype=bool)

    # Pad to K rows
    padded = np.zeros((K, D), dtype=np.float32)
    padded[:m] = cloud.astype(np.float32)
    mask = np.zeros(K, dtype=bool)
    mask[:m] = True

    return padded, mask


def summarize_persistence_diagrams(
    diagrams: list[list[tuple[float, float]]],
    Q: int,
) -> np.ndarray:
    """Compute a fixed-size summary vector from persistence diagrams.

    For each homology degree q = 0, ..., Q, compute 4 statistics:
        1. count:           Number of finite bars
        2. mean_persistence: Mean of (death - birth) for finite bars
        3. max_persistence:  Maximum (death - birth) for finite bars
        4. total_persistence: Sum of (death - birth) for finite bars

    Total dimensionality: 4 × (Q + 1).

    Parameters
    ----------
    diagrams : list of list of (birth, death) tuples
        Q+1 persistence diagrams from M().
    Q : int
        Maximum homology degree.

    Returns
    -------
    summary : np.ndarray, shape (4 * (Q + 1),)
        Flattened summary vector.
    """
    summary = np.zeros(4 * (Q + 1), dtype=np.float32)

    for q in range(min(Q + 1, len(diagrams))):
        dgm = diagrams[q]
        # Filter to finite bars (death ≠ inf)
        finite_bars = [(b, d) for b, d in dgm if np.isfinite(d)]

        offset = q * 4
        summary[offset + 0] = float(len(finite_bars))

        if finite_bars:
            persistences = [d - b for b, d in finite_bars]
            summary[offset + 1] = float(np.mean(persistences))
            summary[offset + 2] = float(np.max(persistences))
            summary[offset + 3] = float(np.sum(persistences))

    return summary


def compute_synapse_features(
    proprio_history: np.ndarray,
    config: ExperimentConfig,
    norm_stats: NormalizationStats,
) -> CachedSynapseFeatures:
    """Compute SYNAPSE features for a single sample.

    Pipeline:
        1. Z-score normalize proprio_history
        2. Call M() with experiment parameters
        3. Pad anchor cloud to K rows
        4. Summarize persistence diagrams

    Parameters
    ----------
    proprio_history : np.ndarray, shape (T, D)
        Raw proprioceptive history for one episode.
    config : ExperimentConfig
        Experiment configuration with SYNAPSE parameters.
    norm_stats : NormalizationStats
        Z-score normalization statistics (from training split).

    Returns
    -------
    CachedSynapseFeatures
    """
    # Step 1: Z-score normalize BEFORE M()
    proprio_norm = norm_stats.normalize(
        np.asarray(proprio_history, dtype=np.float64)
    )

    # Step 2: Call M()
    state = M(
        trajectory=proprio_norm,
        K=config.synapse.K,
        r=config.synapse.r,
        tau=config.synapse.tau,
        weights=config.synapse.weights,
        Q=config.synapse.Q,
        alpha=config.synapse.alpha,
        max_edge_length=config.synapse.max_edge_length,
    )

    # Step 3: Pad anchor cloud
    anchor_feature_dim = config.anchor_feature_dim
    cloud = state.point_cloud

    if cloud.size == 0 or cloud.shape[0] == 0:
        # No anchors produced — create zero-filled cloud with correct dim
        padded_anchors = np.zeros(
            (config.synapse.K, anchor_feature_dim), dtype=np.float32
        )
        anchor_mask = np.zeros(config.synapse.K, dtype=bool)
    else:
        # Ensure the cloud has the expected anchor feature dimension
        # cloud shape is (m, d+3) where d = proprio_dim
        if cloud.shape[1] != anchor_feature_dim:
            raise ValueError(
                f"Anchor cloud dimension mismatch: expected {anchor_feature_dim}, "
                f"got {cloud.shape[1]}. Check that proprio_dim matches the "
                f"trajectory dimensionality."
            )
        padded_anchors, anchor_mask = pad_anchor_cloud(cloud, config.synapse.K)

    # Step 4: Summarize persistence diagrams
    topo_summary = summarize_persistence_diagrams(
        state.persistence_diagrams, config.synapse.Q
    )

    return CachedSynapseFeatures(
        anchors=padded_anchors,
        anchor_mask=anchor_mask,
        topo=topo_summary,
    )


def cache_synapse_features(
    episodes: list[dict],
    config: ExperimentConfig,
    norm_stats: NormalizationStats,
    output_path: Path,
    proprio_key: str = "proprio_history",
) -> dict:
    """Pre-compute M() for all samples in a dataset split and save to disk.

    Parameters
    ----------
    episodes : list of dict
        Each dict must contain `proprio_key` with an ndarray of shape (T, D).
    config : ExperimentConfig
    norm_stats : NormalizationStats
    output_path : Path
        Where to save the .pt file containing cached features.
    proprio_key : str
        Key for accessing proprioceptive history in episode dicts.

    Returns
    -------
    metadata : dict
        Summary statistics about the caching run.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_anchors = []
    all_anchor_masks = []
    all_topo = []
    episode_ids = []
    errors = []

    log.info(
        "Caching SYNAPSE features for %d episodes → %s",
        len(episodes),
        output_path,
    )

    for i, ep in enumerate(episodes):
        ep_id = ep.get("episode_id", f"episode_{i:06d}")
        proprio = ep[proprio_key]
        if isinstance(proprio, torch.Tensor):
            proprio = proprio.numpy()
        proprio = np.asarray(proprio, dtype=np.float64)

        try:
            features = compute_synapse_features(proprio, config, norm_stats)
            all_anchors.append(features.anchors)
            all_anchor_masks.append(features.anchor_mask)
            all_topo.append(features.topo)
            episode_ids.append(ep_id)
        except Exception as e:
            log.error("Failed to compute features for %s: %s", ep_id, e)
            errors.append({"episode_id": ep_id, "error": str(e)})
            # Insert zero-filled placeholders to maintain index alignment
            all_anchors.append(
                np.zeros(
                    (config.synapse.K, config.anchor_feature_dim), dtype=np.float32
                )
            )
            all_anchor_masks.append(np.zeros(config.synapse.K, dtype=bool))
            all_topo.append(np.zeros(config.topo_feature_dim, dtype=np.float32))
            episode_ids.append(ep_id)

    # Stack into tensors
    anchors_tensor = torch.from_numpy(np.stack(all_anchors, axis=0))
    masks_tensor = torch.from_numpy(np.stack(all_anchor_masks, axis=0))
    topo_tensor = torch.from_numpy(np.stack(all_topo, axis=0))

    torch.save(
        {
            "anchors": anchors_tensor,
            "anchor_masks": masks_tensor,
            "topo": topo_tensor,
            "episode_ids": episode_ids,
            "config_snapshot": config.to_dict(),
        },
        output_path,
    )

    metadata = {
        "num_episodes": len(episodes),
        "num_errors": len(errors),
        "errors": errors,
        "anchors_shape": list(anchors_tensor.shape),
        "topo_shape": list(topo_tensor.shape),
        "output_path": str(output_path),
    }

    log.info(
        "Cached %d episodes (%d errors). Anchors: %s, Topo: %s",
        len(episodes),
        len(errors),
        list(anchors_tensor.shape),
        list(topo_tensor.shape),
    )

    return metadata


def load_cached_features(path: Path) -> dict:
    """Load pre-computed SYNAPSE features from disk.

    Parameters
    ----------
    path : Path
        Path to the .pt file saved by cache_synapse_features.

    Returns
    -------
    dict with keys:
        anchors:      torch.Tensor, shape (N, K, D_anchor)
        anchor_masks: torch.Tensor, shape (N, K)
        topo:         torch.Tensor, shape (N, D_topo)
        episode_ids:  list of str
        config_snapshot: dict
    """
    if not path.exists():
        raise FileNotFoundError(f"Cached features not found: {path}")

    data = torch.load(path, weights_only=False)
    log.info(
        "Loaded cached features from %s: %d episodes",
        path,
        data["anchors"].shape[0],
    )
    return data


def verify_cache_consistency(
    episodes: list[dict],
    cached: dict,
    config: ExperimentConfig,
    norm_stats: NormalizationStats,
    max_check: int = 10,
    proprio_key: str = "proprio_history",
    atol: float = 1e-5,
) -> list[str]:
    """Verify cached features match on-the-fly M() computation.

    This is the numerical parity test (§4.5 of PLAN.md): cached features
    must be numerically identical to freshly computed features.

    Parameters
    ----------
    episodes : list of dict
    cached : dict
        Loaded cached features.
    config : ExperimentConfig
    norm_stats : NormalizationStats
    max_check : int
        Maximum number of episodes to verify.
    proprio_key : str
    atol : float
        Absolute tolerance for numerical comparison.

    Returns
    -------
    issues : list of str
        Empty list means all checks passed.
    """
    issues = []
    n_check = min(max_check, len(episodes))

    for i in range(n_check):
        ep = episodes[i]
        ep_id = ep.get("episode_id", f"episode_{i:06d}")

        proprio = ep[proprio_key]
        if isinstance(proprio, torch.Tensor):
            proprio = proprio.numpy()
        proprio = np.asarray(proprio, dtype=np.float64)

        try:
            features = compute_synapse_features(proprio, config, norm_stats)

            if not np.allclose(features.anchors, cached["anchors"][i].numpy(), atol=atol):
                issues.append(f"{ep_id}: anchor cloud mismatch")
            if not np.allclose(features.topo, cached["topo"][i].numpy(), atol=atol):
                issues.append(f"{ep_id}: topological summary mismatch")
            if not np.array_equal(features.anchor_mask, cached["anchor_masks"][i].numpy()):
                issues.append(f"{ep_id}: anchor mask mismatch")
        except Exception as e:
            issues.append(f"{ep_id}: computation failed: {e}")

    if not issues:
        log.info("Cache consistency check passed for %d episodes", n_check)
    else:
        log.warning("Cache consistency issues: %s", issues)

    return issues
