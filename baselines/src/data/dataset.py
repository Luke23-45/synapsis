"""
Unified Data Pipeline — Phase 4 Multi-Dataset Support
======================================================

Provides the core RoboticsDataset, collation, splitting, and DataLoader
creation utilities. Consumes RobotEpisode instances from LeRobot adapters.

All datasets used in baselines are public HuggingFace benchmarks loaded
via the LeRobot adapter layer.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.core.config import Condition, ExperimentConfig
from src.core.normalization import NormalizationStats

# Import the unified RobotEpisode from adapters (single source of truth)
from src.data.adapters.base_adapter import RobotEpisode

class BaselineEpisodeAwareSampler(Sampler[int]):
    """
    Episode-Aware Sampler for Baseline Training.

    Groups samples by episode to maximize cache locality when reading directly
    from HuggingFace/LeRobot datasets (which are often heavily compressed).
    Avoids the random access thrashing of `shuffle=True`.
    """
    def __init__(self, dataset: "RoboticsDataset", shuffle: bool = True, seed: int = 42):
        super().__init__()
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Group flat indices by episode
        self.ep_to_indices = {}
        for flat_idx, (ep_idx, t) in enumerate(dataset._index):
            if ep_idx not in self.ep_to_indices:
                self.ep_to_indices[ep_idx] = []
            self.ep_to_indices[ep_idx].append(flat_idx)
        
        self.ep_list = list(self.ep_to_indices.keys())

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        
        if self.shuffle:
            # Shuffle order of episodes
            eps = rng.permutation(self.ep_list)
        else:
            eps = self.ep_list
            
        final_indices = []
        for ep in eps:
            ep_indices = self.ep_to_indices[ep]
            if self.shuffle:
                # Shuffle frames WITHIN the episode
                ep_indices = rng.permutation(ep_indices).tolist()
            final_indices.extend(ep_indices)
            
        self.epoch += 1
        return iter(final_indices)

    def __len__(self):
        return len(self.dataset._index)

log = logging.getLogger(__name__)


class RoboticsDataset(Dataset):
    """Unified dataset for all experimental conditions.

    Indexes into a list of RobotEpisode instances and produces
    batches suitable for the planner models.

    This dataset is dimension-agnostic: it reads proprio_dim and
    action_dim from the episodes themselves, not from a fixed config.

    Parameters
    ----------
    episodes : list of RobotEpisode
    config : ExperimentConfig
    norm_stats : NormalizationStats
    split : str
        One of "train", "val", "test".
    """

    def __init__(
        self,
        episodes: List[RobotEpisode],
        config: ExperimentConfig,
        norm_stats: NormalizationStats,
        split: str = "train",
    ) -> None:
        self.config = config
        self.norm_stats = norm_stats
        self.split = split
        self.condition = config.condition
        self._episodes = episodes
        self._index: List[Tuple[int, int]] = []

        action_chunk_size = config.data.action_chunk_size
        for ep_idx, ep in enumerate(episodes):
            for t in range(ep.length - action_chunk_size):
                self._index.append((ep_idx, t))

        # Infer dimensions from first episode
        if episodes:
            self._proprio_dim = episodes[0].proprio_history.shape[1]
            self._action_dim = episodes[0].actions.shape[1]
            self._structured_dim = episodes[0].structured_state_dim
        else:
            self._proprio_dim = config.data.proprio_dim
            self._action_dim = config.data.action_dim
            self._structured_dim = config.structured_state_dim

        log.info(
            "RoboticsDataset(%s): %d episodes, %d timesteps, "
            "proprio_dim=%d, action_dim=%d, structured_dim=%d, condition=%s",
            split,
            len(episodes),
            len(self._index),
            self._proprio_dim,
            self._action_dim,
            self._structured_dim,
            self.condition.value,
        )

    @property
    def proprio_dim(self) -> int:
        return self._proprio_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def structured_state_dim(self) -> int:
        return self._structured_dim

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, t = self._index[idx]
        ep = self._episodes[ep_idx]
        history_end = t + 1
        if self.condition == Condition.A1_RECENT:
            history_start = max(0, history_end - self.config.data.history_window)
        else:
            history_start = 0

        proprio_history_np = ep.proprio_history[history_start:history_end]
        structured_history_np = ep.structured_history[history_start:history_end]

        proprio = torch.as_tensor(ep.proprio_history[t], dtype=torch.float32)
        proprio_history = torch.as_tensor(proprio_history_np, dtype=torch.float32)
        structured_state = torch.as_tensor(ep.structured_history[t], dtype=torch.float32)
        structured_history = torch.as_tensor(structured_history_np, dtype=torch.float32)
        action_chunk = torch.as_tensor(
            ep.actions[t : t + self.config.data.action_chunk_size],
            dtype=torch.float32,
        )
        phase_label = torch.tensor(int(ep.gt_phase[t]), dtype=torch.long)
        episode_length = torch.tensor(ep.length, dtype=torch.long)
        timestep = torch.tensor(t, dtype=torch.long)
        history_length = torch.tensor(proprio_history.shape[0], dtype=torch.long)

        sample = {
            "proprio": proprio,
            "proprio_history": proprio_history,
            "structured_state": structured_state,
            "structured_history": structured_history,
            "action_chunk": action_chunk,
            "phase_label": phase_label,
            "episode_length": episode_length,
            "timestep": timestep,
            "history_length": history_length,
            "episode_idx": torch.tensor(ep_idx, dtype=torch.long),
            "dataset_sample_idx": torch.tensor(idx, dtype=torch.long),
        }

        if self.condition.uses_synapse:
            if (
                ep.synapse_anchors is not None
                and ep.synapse_topo is not None
            ):
                sample["synapse_anchors"] = torch.from_numpy(
                    ep.synapse_anchors.copy()
                ).float()
                sample["synapse_topo"] = torch.from_numpy(
                    ep.synapse_topo.copy()
                ).float()
            else:
                sample["synapse_anchors"] = torch.zeros(
                    self.config.synapse.K,
                    self.config.anchor_feature_dim,
                    dtype=torch.float32,
                )
                sample["synapse_topo"] = torch.zeros(
                    self.config.topo_feature_dim,
                    dtype=torch.float32,
                )

        return sample


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function that handles variable-length proprio/structured histories.

    Pads all history sequences to the maximum length in the batch,
    using first-observation replication (standard in robotics IL).
    """
    max_hist_len = max(int(s["proprio_history"].shape[0]) for s in batch)
    batch_size = len(batch)
    proprio_dim = batch[0]["proprio_history"].shape[1]
    structured_dim = batch[0]["structured_history"].shape[1]
    padded_histories = torch.empty(
        batch_size,
        max_hist_len,
        proprio_dim,
        dtype=batch[0]["proprio_history"].dtype,
    )
    padded_structured_histories = torch.empty(
        batch_size,
        max_hist_len,
        structured_dim,
        dtype=batch[0]["structured_history"].dtype,
    )

    for row, sample in enumerate(batch):
        hist = sample["proprio_history"]
        struct = sample["structured_history"]
        hist_len = hist.shape[0]
        if hist_len < max_hist_len:
            pad_len = max_hist_len - hist_len
            padded_histories[row, :pad_len] = hist[:1, :].expand(pad_len, -1)
            padded_histories[row, pad_len:] = hist
            padded_structured_histories[row, :pad_len] = struct[:1, :].expand(pad_len, -1)
            padded_structured_histories[row, pad_len:] = struct
        else:
            padded_histories[row] = hist
            padded_structured_histories[row] = struct

    def _stack_optional_long(name: str, default_fn) -> torch.Tensor:
        values = []
        for sample in batch:
            if name in sample:
                values.append(sample[name])
            else:
                values.append(torch.tensor(default_fn(sample), dtype=torch.long))
        return torch.stack(values)

    result = {
        "proprio": torch.stack([s["proprio"] for s in batch]),
        "proprio_history": padded_histories,
        "structured_state": torch.stack(
            [s["structured_state"] for s in batch]
        ),
        "structured_history": padded_structured_histories,
        "action_chunk": torch.stack([s["action_chunk"] for s in batch]),
        "phase_label": torch.stack([s["phase_label"] for s in batch]),
        "episode_length": torch.stack(
            [s["episode_length"] for s in batch]
        ),
        "timestep": torch.stack([s["timestep"] for s in batch]),
        "history_length": _stack_optional_long(
            "history_length",
            lambda sample: int(sample["proprio_history"].shape[0]),
        ),
        "episode_idx": _stack_optional_long("episode_idx", lambda sample: -1),
        "dataset_sample_idx": _stack_optional_long("dataset_sample_idx", lambda sample: -1),
    }

    if "synapse_anchors" in batch[0]:
        result["synapse_anchors"] = torch.stack(
            [s["synapse_anchors"] for s in batch]
        )
        result["synapse_topo"] = torch.stack(
            [s["synapse_topo"] for s in batch]
        )

    return result


def split_episodes(
    episodes: List[RobotEpisode],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[RobotEpisode], List[RobotEpisode], List[RobotEpisode]]:
    """Split episodes into train/val/test sets with reproducible shuffling."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(episodes))
    rng.shuffle(indices)

    n = len(episodes)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_eps = [episodes[i] for i in indices[:n_train]]
    val_eps = [episodes[i] for i in indices[n_train : n_train + n_val]]
    test_eps = [episodes[i] for i in indices[n_train + n_val :]]

    log.info(
        "Split %d episodes: train=%d, val=%d, test=%d",
        n,
        len(train_eps),
        len(val_eps),
        len(test_eps),
    )
    return train_eps, val_eps, test_eps


def create_dataloaders(
    train_eps: List[RobotEpisode],
    val_eps: List[RobotEpisode],
    test_eps: List[RobotEpisode],
    config: ExperimentConfig,
    norm_stats: NormalizationStats,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from episode lists.

    Each DataLoader uses the unified collate_fn that handles
    variable-length histories across all dataset types.
    """
    train_dataset = RoboticsDataset(
        train_eps, config, norm_stats, split="train"
    )
    val_dataset = RoboticsDataset(
        val_eps, config, norm_stats, split="val"
    )
    test_dataset = RoboticsDataset(
        test_eps, config, norm_stats, split="test"
    )

    loader_kwargs = {
        "num_workers": config.training.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": config.training.pin_memory,
    }
    if config.training.num_workers > 0:
        loader_kwargs["persistent_workers"] = config.training.persistent_workers
        loader_kwargs["prefetch_factor"] = config.training.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=BaselineEpisodeAwareSampler(train_dataset, shuffle=True, seed=config.seed),
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader
