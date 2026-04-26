# FILE: experiments/end_to_end/data/synapse_e2e_dataset.py
# (Adapted from working/m1/semantic_planner_dataset.py for Z2 E2E Training)

"""
SYNAPSE Z2 End-to-End Dataset (Adapted from Semantic Planner Dataset v9.1).

This module defines the high-performance, standalone data pipeline for the Z2
end-to-end training loop. It reads directly from the LMDB dataset using the
ExpertTrajectoryDataset reader and produces training samples containing:

  - Full prefix proprioceptive history (variable-length, for EventEncoder)
  - Current proprioceptive state
  - Future action chunk (fixed-length, for ActionHead prediction)
  - Semantic phase label (for diagnostics / auxiliary supervision)

Key Architectural Differences from SemanticPlannerDataset:
----------------------------------------------------------
1. **No Images**: Z2 operates on proprioceptive state sequences only.
   The event encoder, selector, and memory operator all take (T, state_dim).

2. **Full Prefix History**: Instead of a fixed observation horizon (e.g., 2 frames),
   Z2 requires the ENTIRE trajectory prefix [0, t] as input. The EventEncoder
   processes the full history to compute saliency scores at every timestep.

3. **Action Targets (not Pose Targets)**: Z2 predicts raw action chunks
   (T, action_dim=8) rather than end-effector pose trajectories.

4. **Intelligent Terminal Padding**: Like the semantic planner, we pad action
   chunks at the end of episodes with the last valid action, teaching the
   model to predict "hold" behavior.

Dependencies:
    - experiments.empirical.dataset.expert_dataset (ExpertTrajectoryDataset)
    - torch, numpy
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class SynapseE2EDataset(Dataset):
    """
    High-performance dataset for Z2 end-to-end training.

    Reads directly from LMDB via ExpertTrajectoryDataset, producing samples
    containing full prefix history + future action chunks.

    Attributes:
        dataset_path (str): Path to the .lmdb expert demonstration dataset.
        chunk_size (int): Number of future actions to predict (action chunking).
        min_history (int): Minimum history length required (≥2 for EventEncoder diffs).
        proprio_noise (float): Noise std for proprioceptive augmentation (DAgger-lite).
        use_aug (bool): Whether to apply proprioceptive noise augmentation.
    """

    def __init__(
        self,
        dataset_path: str,
        chunk_size: int = 10,
        min_history: int = 2,
        proprio_noise: float = 0.005,
        use_aug: bool = False,
    ):
        super().__init__()

        log.info("=" * 60)
        log.info("INITIALIZING SYNAPSE E2E DATASET")
        log.info("  > Dataset Path:   %s", dataset_path)
        log.info("  > Chunk Size:     %d steps (Action Prediction)", chunk_size)
        log.info("  > Min History:    %d frames", min_history)
        log.info("  > Proprio Noise:  %s (DAgger-lite)", proprio_noise)
        log.info("  > Augmentations:  %s", use_aug)
        log.info("=" * 60)

        self.dataset_path = dataset_path
        self.chunk_size = chunk_size
        self.min_history = max(min_history, 2)  # EventEncoder needs ≥2 for diffs
        self.proprio_noise = proprio_noise
        self.use_aug = use_aug

        # --- 1. Load Data ---
        from experiments.end_to_end.data.e2e_expert_dataset import ExpertTrajectoryDataset
        self.expert_reader = ExpertTrajectoryDataset(
            demo_path=dataset_path,
            observation_horizon=2,       # Minimal, we handle history ourselves
            action_horizon=chunk_size,
        )

        # --- 2. Validate Dataset ---
        self._validate_dataset()

        # --- 3. Build Explicit Sample Index ---
        # Each sample = (episode_idx, timestep_t)
        # Valid timesteps: t >= min_history - 1 (enough prefix frames)
        # We index up to the very end of the episode — future action chunks
        # at the boundary are padded with the terminal action.
        self.samples: List[Tuple[int, int]] = []
        total_episodes = self.expert_reader.get_num_episodes()

        for ep_idx in range(total_episodes):
            ep_len = self.expert_reader.get_episode_length(ep_idx)
            start_t = self.min_history - 1  # Need ≥ min_history frames in history
            end_t = ep_len - 1              # Include last frame (padding handles actions)

            if start_t <= end_t:
                for t in range(start_t, end_t + 1):
                    self.samples.append((ep_idx, t))
            else:
                log.warning(
                    "Episode %d too short (len=%d) for min_history=%d. Skipping.",
                    ep_idx, ep_len, self.min_history,
                )

        log.info("Index built successfully.")
        log.info("  > Total Episodes: %d", total_episodes)
        log.info("  > Total Samples:  %d", len(self.samples))

        # --- 4. Cache episode metadata for fast access ---
        self._episode_lengths = [
            self.expert_reader.get_episode_length(i) for i in range(total_episodes)
        ]

    def _validate_dataset(self) -> None:
        """Check that the LMDB has required modalities for Z2 training."""
        if self.expert_reader.get_num_episodes() == 0:
            raise RuntimeError("Dataset is empty! No episodes found.")

        first_ep_meta = self.expert_reader.episode_metadata[0]
        modalities = first_ep_meta["modalities"]

        required = ["proprio", "actions", "gt_phase"]
        missing = [k for k in required if k not in modalities]
        if missing:
            raise RuntimeError(
                f"Dataset missing required modalities for Z2 training: {missing}. "
                "Required: proprio, actions, gt_phase."
            )

        # Log available modalities
        log.info("  Available modalities: %s", list(modalities.keys()))
        proprio_shape = modalities["proprio"]["shape"]
        action_shape = modalities["actions"]["shape"]
        log.info("  Proprio dim: %d, Action dim: %d", proprio_shape[-1], action_shape[-1])

    def __len__(self) -> int:
        return len(self.samples)

    def get_phase_label(self, idx: int) -> int:
        """Fast access to phase label without loading full modalities.

        Useful for computing class distributions for WeightedRandomSampler.
        """
        ep_idx, t = self.samples[idx]
        ep_meta = self.expert_reader.episode_metadata[ep_idx]
        phase_meta = ep_meta["modalities"]["gt_phase"]
        phase_array = self.expert_reader._get_full_modality_array(
            key=phase_meta["key"],
            compression=phase_meta["compression"],
            dtype_str=phase_meta["dtype"],
            shape_list=tuple(phase_meta["shape"]),
        )
        return int(phase_array[t])

    def _get_action_chunk(
        self,
        current_t: int,
        all_actions: np.ndarray,
    ) -> np.ndarray:
        """Extract future action chunk with intelligent terminal padding.

        If the episode ends before the full chunk is filled, remaining steps
        are padded with the last valid action. This teaches the model to
        predict "hold" behavior at episode boundaries.

        Parameters
        ----------
        current_t : int
            Current timestep.
        all_actions : np.ndarray
            Shape (T, action_dim). Full episode actions.

        Returns
        -------
        np.ndarray
            Shape (chunk_size, action_dim).
        """
        episode_len = len(all_actions)
        start_t = current_t
        end_t = min(start_t + self.chunk_size, episode_len)

        if start_t < episode_len:
            chunk = all_actions[start_t:end_t].copy()
        else:
            chunk = np.empty((0, all_actions.shape[1]), dtype=all_actions.dtype)

        pad_len = self.chunk_size - len(chunk)
        if pad_len > 0:
            terminal_action = all_actions[-1]
            padding = np.tile(terminal_action, (pad_len, 1))
            chunk = np.concatenate([chunk, padding], axis=0)

        return chunk.astype(np.float32)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Construct a complete Z2 training sample.

        Returns
        -------
        Dict with:
            - structured_history: (t+1, state_dim) float32 — full prefix
            - structured_state: (state_dim,) float32 — current state
            - ground_truth_actions: (chunk_size, action_dim) float32
            - phase_label: long scalar
            - history_length: long scalar — actual history length
            - episode_idx: long scalar
            - timestep: long scalar
        """
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range.")

        try:
            ep_idx, t = self.samples[idx]
            ep_meta = self.expert_reader.episode_metadata[ep_idx]

            # --- Load modalities (leverages LRU cache) ---
            def get_mod(name: str) -> np.ndarray:
                meta = ep_meta["modalities"][name]
                return self.expert_reader._get_full_modality_array(
                    key=meta["key"],
                    compression=meta["compression"],
                    dtype_str=meta["dtype"],
                    shape_list=tuple(meta["shape"]),
                )

            all_proprio = get_mod("proprio")    # (T, 22)
            all_actions = get_mod("actions")    # (T, 8)
            all_phases = get_mod("gt_phase")    # (T, 1) or (T,)

            # --- Full prefix history [0, t+1) ---
            history = all_proprio[: t + 1].copy()  # (t+1, state_dim)

            # --- Current state ---
            current_state = all_proprio[t].copy()  # (state_dim,)

            # --- Proprioceptive noise injection (DAgger-lite) ---
            if self.use_aug and self.proprio_noise > 0:
                noise = np.random.normal(
                    0, self.proprio_noise, size=current_state.shape
                ).astype(np.float32)
                current_state = current_state + noise

            # --- Future action chunk ---
            action_chunk = self._get_action_chunk(t, all_actions)

            # --- Phase label ---
            phase_raw = all_phases[t]
            phase_val = int(phase_raw.item()) if hasattr(phase_raw, "item") else int(phase_raw)

            return {
                "structured_history": torch.from_numpy(history.astype(np.float32)),
                "structured_state": torch.from_numpy(current_state.astype(np.float32)),
                "ground_truth_actions": torch.from_numpy(action_chunk),
                "phase_label": torch.tensor(phase_val, dtype=torch.long),
                "history_length": torch.tensor(t + 1, dtype=torch.long),
                "episode_idx": torch.tensor(ep_idx, dtype=torch.long),
                "timestep": torch.tensor(t, dtype=torch.long),
            }

        except Exception as e:
            log.error(
                "Error loading sample %d (Ep %d, T %d): %s",
                idx, self.samples[idx][0], self.samples[idx][1], e,
                exc_info=False,
            )
            return None
