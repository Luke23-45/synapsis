"""
PyTorch Dataset for Z2 end-to-end training.

Loads episodes from the AppliedDataset (LMDB or CSV) and produces
(episode, timestep) samples suitable for the SynapseEndToEndModel.

Each sample provides:
  - structured_history: full prefix trajectory up to timestep t
  - structured_state: current state at timestep t
  - ground_truth_actions: chunk of future actions starting at t
  - phase_label: ground-truth phase for diagnostics
  - episode_idx / timestep: metadata for analysis

Design decisions:
  1. Full prefix history (no fixed window) — Z2 operates on variable-length
  2. Uniform random sampling over all valid (episode, timestep) pairs
  3. Optional state normalization via pre-computed statistics
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalization import NormalizationStats

log = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    """PyTorch Dataset producing (episode, timestep) samples for Z2 training.

    Parameters
    ----------
    episodes : List[dict]
        List of episode dicts, each containing at minimum:
          - "states": np.ndarray (T, state_dim), float32
          - "actions": np.ndarray (T, action_dim), float32
          - "phase_labels": np.ndarray (T,), int64
    action_chunk_size : int
        Number of future actions to predict from each timestep.
    norm_stats : NormalizationStats, optional
        If provided, states are normalized: (s - mean) / std.
    state_key : str
        Key for the state array in each episode dict.
    action_key : str
        Key for the action array in each episode dict.
    phase_key : str
        Key for the phase label array in each episode dict.
    """

    def __init__(
        self,
        episodes: List[dict],
        action_chunk_size: int = 10,
        norm_stats: Optional[NormalizationStats] = None,
        state_key: str = "states",
        action_key: str = "actions",
        phase_key: str = "phase_labels",
    ) -> None:
        super().__init__()
        self.episodes = episodes
        self.action_chunk_size = action_chunk_size
        self.norm_stats = norm_stats
        self.state_key = state_key
        self.action_key = action_key
        self.phase_key = phase_key

        # Build index of valid (episode_idx, timestep) pairs.
        # A timestep t is valid if there are at least action_chunk_size
        # actions remaining: t <= T - action_chunk_size.
        # We also require t >= 1 so the history has at least 2 frames
        # (needed for the event encoder's diff computation).
        self._index: List[Tuple[int, int]] = []
        for ep_idx, ep in enumerate(episodes):
            T = len(ep[state_key])
            # Valid range: t in [1, T - action_chunk_size]
            max_t = T - action_chunk_size
            for t in range(1, max_t + 1):
                self._index.append((ep_idx, t))

        if not self._index:
            raise ValueError(
                f"No valid samples found. {len(episodes)} episodes with "
                f"action_chunk_size={action_chunk_size}. Check episode lengths."
            )

        log.info(
            "TrajectoryDataset: %d episodes → %d valid samples "
            "(action_chunk_size=%d)",
            len(episodes),
            len(self._index),
            action_chunk_size,
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, t = self._index[idx]
        ep = self.episodes[ep_idx]

        states = ep[self.state_key]   # (T, state_dim) float32
        actions = ep[self.action_key]  # (T, action_dim) float32

        # Structured history: full prefix [0, t+1) — includes current state
        history = states[: t + 1].copy()  # (t+1, state_dim)

        # Current state
        current_state = states[t].copy()  # (state_dim,)

        # Ground-truth action chunk
        gt_actions = actions[t : t + self.action_chunk_size].copy()  # (chunk, action_dim)

        # Phase label at current timestep (for diagnostics)
        phase_label = int(ep[self.phase_key][t])

        # Apply normalization if available
        if self.norm_stats is not None:
            mean = self.norm_stats.state_mean
            std = self.norm_stats.state_std
            history = (history - mean) / std
            current_state = (current_state - mean) / std

        return {
            "structured_history": torch.from_numpy(history.astype(np.float32)),
            "structured_state": torch.from_numpy(current_state.astype(np.float32)),
            "ground_truth_actions": torch.from_numpy(gt_actions.astype(np.float32)),
            "phase_label": torch.tensor(phase_label, dtype=torch.long),
            "episode_idx": torch.tensor(ep_idx, dtype=torch.long),
            "timestep": torch.tensor(t, dtype=torch.long),
        }

    @property
    def state_dim(self) -> int:
        """Dimension of the state vector."""
        return self.episodes[0][self.state_key].shape[1]

    @property
    def action_dim(self) -> int:
        """Dimension of the action vector."""
        return self.episodes[0][self.action_key].shape[1]

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)
