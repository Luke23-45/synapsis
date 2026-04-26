"""
Dataset-level normalization statistics for the Z2 end-to-end pipeline.

Computes per-feature mean and std from training episodes, used for:
1. Setting NormalizedLift.mu / sigma (anchor normalization — §8 of 02_rigorous_architecture.md)
2. Optionally standardizing input states before the event encoder

All statistics are computed on the *training set only* to prevent data leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

log = logging.getLogger(__name__)


@dataclass
class NormalizationStats:
    """Per-feature normalization statistics computed from training data.

    Attributes
    ----------
    state_mean : np.ndarray
        Shape (state_dim,). Per-feature mean of proprio states.
    state_std : np.ndarray
        Shape (state_dim,). Per-feature std of proprio states.
        Guaranteed > 0 (clamped to 1e-6 floor).
    anchor_mu : np.ndarray
        Shape (state_dim + 3,). Mean for NormalizedLift (includes t, delta, xi).
    anchor_sigma : np.ndarray
        Shape (state_dim + 3,). Std for NormalizedLift.
        Guaranteed > 0 (clamped to 1e-6 floor).
    num_episodes : int
        Number of training episodes used to compute statistics.
    num_timesteps : int
        Total number of timesteps across all training episodes.
    """

    state_mean: np.ndarray
    state_std: np.ndarray
    anchor_mu: np.ndarray
    anchor_sigma: np.ndarray
    num_episodes: int
    num_timesteps: int

    def state_mean_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.state_mean).float()

    def state_std_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.state_std).float()

    def anchor_mu_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.anchor_mu).float()

    def anchor_sigma_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.anchor_sigma).float()


def compute_normalization_stats(
    episodes: List[dict],
    state_key: str = "states",
    eps: float = 1e-6,
) -> NormalizationStats:
    """Compute normalization statistics from a list of training episodes.

    Each episode dict must have:
      - ``states``: np.ndarray of shape (T, state_dim), float32
      - ``actions``: np.ndarray of shape (T, action_dim), float32

    The anchor vector v(a_j) = [t_j, s_j^T, delta_j, xi_j] has dimension
    state_dim + 3. Since we don't have ground-truth event scores at this
    stage, we approximate the anchor statistics using:
      - t: uniform in [0, 1]
      - s: proprio states
      - delta: inter-event gaps (approximated as 1.0 for dense vectors)
      - xi: event scores (approximated as 0.0 placeholder, learned at runtime)

    Parameters
    ----------
    episodes : List[dict]
        Training episodes with "states" arrays.
    state_key : str
        Key for the state array in each episode dict.
    eps : float
        Minimum std floor to prevent division by zero.

    Returns
    -------
    NormalizationStats
    """
    if not episodes:
        raise ValueError("Cannot compute normalization from empty episode list")

    # Collect all states
    all_states = []
    total_T = 0
    for ep in episodes:
        states = ep[state_key]
        all_states.append(states)
        total_T += len(states)

    all_states_cat = np.concatenate(all_states, axis=0)  # (N, state_dim)
    state_dim = all_states_cat.shape[1]

    # State statistics
    state_mean = all_states_cat.mean(axis=0).astype(np.float32)
    state_std = all_states_cat.std(axis=0).astype(np.float32)
    state_std = np.maximum(state_std, eps)

    # Anchor vector statistics:
    # v(a_j) = [t_j, s_j^T, delta_j, xi_j]  ∈ R^{state_dim + 3}
    #
    # We approximate these from the dense training data:
    #   t: normalized timestamps ~Uniform(0, 1) → mean=0.5, std≈0.29
    #   s: proprio states → use state_mean, state_std
    #   delta: inter-event gaps → approximate with 1.0 mean, 1.0 std
    #   xi: event scores → will be learned; use 0.0 mean, 1.0 std
    #
    # These are starting values. During training, the NormalizedLift
    # can be re-fitted if needed (the set_normalization method exists).

    anchor_mu = np.zeros(state_dim + 3, dtype=np.float32)
    anchor_sigma = np.ones(state_dim + 3, dtype=np.float32)

    # t dimension (index 0): mean=0.5, std≈0.29 (uniform distribution)
    anchor_mu[0] = 0.5
    anchor_sigma[0] = max(0.29, eps)

    # s dimensions (indices 1:state_dim+1): use state statistics
    anchor_mu[1 : state_dim + 1] = state_mean
    anchor_sigma[1 : state_dim + 1] = state_std

    # delta dimension (index state_dim+1): inter-event gap
    # Approximate from training data: mean gap between events
    anchor_mu[state_dim + 1] = 1.0
    anchor_sigma[state_dim + 1] = max(1.0, eps)

    # xi dimension (index state_dim+2): event score (learned)
    anchor_mu[state_dim + 2] = 0.0
    anchor_sigma[state_dim + 2] = max(1.0, eps)

    anchor_sigma = np.maximum(anchor_sigma, eps)

    stats = NormalizationStats(
        state_mean=state_mean,
        state_std=state_std,
        anchor_mu=anchor_mu,
        anchor_sigma=anchor_sigma,
        num_episodes=len(episodes),
        num_timesteps=total_T,
    )

    log.info(
        "Normalization stats: %d episodes, %d timesteps, state_dim=%d, "
        "anchor_dim=%d",
        stats.num_episodes,
        stats.num_timesteps,
        state_dim,
        state_dim + 3,
    )

    return stats
