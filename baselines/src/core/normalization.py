"""
Z-Score Pre-Normalization — Professor Feedback #3
==================================================

SYNAPSE uses geometric and topological math (persistence diagrams, Euclidean
distances) on the state space. If the proprioception array contains mixed
units — joint angles in [-3.14, 3.14] alongside gripper aperture in [0, 0.05] —
the larger-magnitude dimensions dominate distance metrics, effectively
blinding M() to the gripper.

Normalization MUST happen before M() is applied, not just inside the neural
network. This module computes and applies Z-score normalization using
statistics derived from the training split only (no data leakage).

Protocol (§3.7 of PLAN.md):
    1. Compute global mean μ and std σ for all proprioceptive dimensions
       across the training split only.
    2. Apply Z-score: x_norm = (x - μ) / σ to raw proprio_history BEFORE
       passing to synapse_core.M().
    3. Store μ and σ as dataset metadata; apply the same transform to
       validation and test splits.
    4. The neural network's proprio encoder receives the same normalized input.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)

_EPSILON = 1e-8


class NormalizationStats:
    """Immutable Z-score normalization statistics computed from training data.

    Attributes
    ----------
    mean : np.ndarray, shape (D,)
        Per-dimension mean computed from the training split.
    std : np.ndarray, shape (D,)
        Per-dimension standard deviation computed from the training split.
    dim : int
        Dimensionality of the proprioceptive vector.
    """

    __slots__ = ("mean", "std", "dim")

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        if mean.shape != std.shape:
            raise ValueError(
                f"Shape mismatch: mean {mean.shape} vs std {std.shape}"
            )
        if mean.ndim != 1:
            raise ValueError(f"Expected 1-D arrays, got {mean.ndim}-D")

        self.mean = mean.astype(np.float64)
        self.std = std.astype(np.float64)
        self.dim = mean.shape[0]

        # Guard against zero std (constant dimensions)
        zero_std_mask = self.std < _EPSILON
        if np.any(zero_std_mask):
            log.warning(
                "Dimensions with near-zero std detected: %s. "
                "Setting std to 1.0 to avoid division by zero.",
                np.where(zero_std_mask)[0].tolist(),
            )
            self.std[zero_std_mask] = 1.0

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Apply Z-score normalization: (x - μ) / σ.

        Parameters
        ----------
        x : np.ndarray, shape (..., D)
            Input proprioceptive data. Last dimension must match `self.dim`.

        Returns
        -------
        np.ndarray, shape (..., D)
            Normalized data with zero mean and unit variance per dimension.
        """
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Last dimension mismatch: expected {self.dim}, got {x.shape[-1]}"
            )
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Reverse Z-score normalization: x * σ + μ.

        Parameters
        ----------
        x : np.ndarray, shape (..., D)
            Normalized data.

        Returns
        -------
        np.ndarray, shape (..., D)
            Data in original scale.
        """
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Last dimension mismatch: expected {self.dim}, got {x.shape[-1]}"
            )
        return x * self.std + self.mean

    def normalize_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Z-score normalization to a PyTorch tensor.

        Parameters
        ----------
        x : torch.Tensor, shape (..., D)

        Returns
        -------
        torch.Tensor, shape (..., D)
        """
        mean_t = torch.from_numpy(self.mean).to(x.device, dtype=x.dtype)
        std_t = torch.from_numpy(self.std).to(x.device, dtype=x.dtype)
        return (x - mean_t) / std_t

    def save(self, path: Path) -> None:
        """Save normalization statistics to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"mean": torch.from_numpy(self.mean), "std": torch.from_numpy(self.std)},
            path,
        )
        log.info("Saved normalization stats to %s", path)

    @classmethod
    def load(cls, path: Path) -> NormalizationStats:
        """Load normalization statistics from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {path}")
        data = torch.load(path, weights_only=True)
        return cls(
            mean=data["mean"].numpy(),
            std=data["std"].numpy(),
        )

    def __repr__(self) -> str:
        return (
            f"NormalizationStats(dim={self.dim}, "
            f"mean_range=[{self.mean.min():.4f}, {self.mean.max():.4f}], "
            f"std_range=[{self.std.min():.4f}, {self.std.max():.4f}])"
        )


def compute_normalization_stats(
    proprio_arrays: list[np.ndarray],
) -> NormalizationStats:
    """Compute Z-score normalization statistics from training data only.

    This is the sole entry point for deriving normalization statistics.
    It MUST be called on the training split only — never on validation
    or test data — to prevent data leakage.

    Parameters
    ----------
    proprio_arrays : list of np.ndarray, each shape (T_i, D)
        Proprioceptive histories from the training split. Episodes may
        have different lengths T_i but must share dimensionality D.

    Returns
    -------
    NormalizationStats
        Immutable statistics object ready for normalization/denormalization.
    """
    if not proprio_arrays:
        raise ValueError("Cannot compute stats from empty list")

    # Concatenate along time axis: shape (Σ T_i, D)
    all_proprio = np.concatenate(proprio_arrays, axis=0)

    if all_proprio.ndim != 2:
        raise ValueError(
            f"Expected 2-D concatenated array, got {all_proprio.ndim}-D"
        )

    dim = all_proprio.shape[1]
    log.info(
        "Computing normalization stats from %d timesteps × %d dimensions",
        all_proprio.shape[0],
        dim,
    )

    mean = all_proprio.mean(axis=0)
    std = all_proprio.std(axis=0)

    stats = NormalizationStats(mean=mean, std=std)
    log.info("Normalization stats: %s", stats)
    return stats


def compute_normalization_stats_from_episodes(
    episodes: list[dict],
    proprio_key: str = "proprio_history",
) -> NormalizationStats:
    """Convenience: compute stats from a list of episode dicts.

    Parameters
    ----------
    episodes : list of dict
        Each dict must contain `proprio_key` with an ndarray of shape (T, D).
    proprio_key : str
        Key to access the proprioceptive history array.

    Returns
    -------
    NormalizationStats
    """
    proprio_arrays = []
    for ep in episodes:
        arr = ep[proprio_key]
        if isinstance(arr, torch.Tensor):
            arr = arr.numpy()
        proprio_arrays.append(np.asarray(arr, dtype=np.float64))

    return compute_normalization_stats(proprio_arrays)
