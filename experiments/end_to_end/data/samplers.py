# FILE: experiments/end_to_end/data/samplers.py
# (Adapted from working/m1/samplers.py for Z2 E2E Training)

"""
Episode-Aware Sampler for Z2 End-to-End Training.

Adapted from the Semantic Planner's SOTA sampler. Key features:

1. **Cache Locality**: Yields all indices of Episode N before Episode M,
   maximizing LMDB read-ahead and LRU cache hits.
2. **DDP Support**: Automatically partitions episodes across GPUs/ranks.
3. **Vectorized Init**: Uses NumPy for O(1) index construction.
4. **Subset-Safe**: Handles PyTorch Subsets (from random_split) seamlessly.

For Z2 training, cache locality is critical because each sample requires
loading the FULL prefix history for the episode. By grouping samples from
the same episode together, we avoid re-reading the same LMDB keys.
"""

from __future__ import annotations

import logging
import math
from typing import Iterator, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler, Subset

log = logging.getLogger(__name__)


class EpisodeAwareSampler(Sampler[int]):
    """
    SOTA Sampler that yields samples grouped by episode for cache locality.

    For Z2 training, this is critical because:
    - Each sample loads the full prefix history from LMDB
    - The ExpertTrajectoryDataset uses LRU caching per-episode
    - Grouping samples by episode maximizes cache hits

    Parameters
    ----------
    dataset : Sized
        Must expose ``expert_reader`` with ``_cumulative_chunks`` attribute,
        OR must be a ``SynapseE2EDataset`` with ``samples`` attribute.
    shuffle : bool
        Whether to shuffle episode order each epoch.
    seed : int
        Base random seed for reproducibility.
    num_replicas : int, optional
        Number of DDP processes. Auto-detected if None.
    rank : int, optional
        Current DDP rank. Auto-detected if None.
    drop_last : bool
        Whether to drop trailing episodes that don't divide evenly.
    """

    def __init__(
        self,
        dataset,
        shuffle: bool = True,
        seed: int = 42,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = False,
    ):
        super().__init__()

        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # --- 1. Resolve DDP Parameters ---
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                num_replicas = 1
                rank = 0
            else:
                num_replicas = dist.get_world_size()
                rank = dist.get_rank()

        if rank is None:
            rank = 0

        self.num_replicas = num_replicas
        self.rank = rank

        if self.rank == 0:
            log.info(
                "EpisodeAwareSampler: DDP=%s, Rank=%d/%d",
                self.num_replicas > 1, self.rank, self.num_replicas,
            )

        # --- 2. Resolve Underlying Data Structure ---
        if isinstance(dataset, Subset):
            self.subset_indices = np.array(dataset.indices)
            full_dataset = dataset.dataset
        else:
            self.subset_indices = np.arange(len(dataset))
            full_dataset = dataset

        # --- 3. Build Episode-to-Samples Mapping ---
        # SynapseE2EDataset stores samples as (ep_idx, t) tuples
        if hasattr(full_dataset, "samples"):
            self._build_from_samples_list(full_dataset, self.subset_indices)
        elif hasattr(full_dataset, "expert_reader"):
            self._build_from_cumulative_chunks(full_dataset, self.subset_indices)
        else:
            raise ValueError(
                "Dataset must expose 'samples' (SynapseE2EDataset) or "
                "'expert_reader' (ExpertTrajectoryDataset) for episode-aware sampling."
            )

        # --- 4. Calculate DDP Lengths ---
        total_episodes = len(self.available_episodes)
        if self.drop_last and self.num_replicas > 1:
            self.num_episodes_per_replica = math.floor(total_episodes / self.num_replicas)
        else:
            self.num_episodes_per_replica = math.ceil(total_episodes / self.num_replicas)
        self.total_size_episodes = self.num_episodes_per_replica * self.num_replicas

    def _build_from_samples_list(self, dataset, subset_indices: np.ndarray) -> None:
        """Build episode map from SynapseE2EDataset.samples list."""
        # Each sample in dataset.samples is (ep_idx, timestep)
        # We need to map global dataset indices → episode IDs
        episode_map: dict[int, list[int]] = {}

        for global_idx in subset_indices:
            ep_idx, _ = dataset.samples[global_idx]
            if ep_idx not in episode_map:
                episode_map[ep_idx] = []
            episode_map[ep_idx].append(int(global_idx))

        self.episode_map = {k: np.array(v) for k, v in episode_map.items()}
        self.available_episodes = list(self.episode_map.keys())

    def _build_from_cumulative_chunks(self, dataset, subset_indices: np.ndarray) -> None:
        """Build episode map from ExpertTrajectoryDataset cumulative chunks."""
        cumulative_chunks = np.array(dataset.expert_reader._cumulative_chunks)

        # Vectorized: find which episode every index belongs to
        episode_assignments = np.searchsorted(
            cumulative_chunks, subset_indices, side="right"
        )

        # Group by episode using sorting
        sort_order = np.argsort(episode_assignments)
        sorted_indices = subset_indices[sort_order]
        sorted_episodes = episode_assignments[sort_order]

        unique_eps, split_indices = np.unique(sorted_episodes, return_index=True)
        grouped_indices = np.split(sorted_indices, split_indices[1:])

        self.episode_map = {
            int(ep_id): indices
            for ep_id, indices in zip(unique_eps, grouped_indices)
            if len(indices) > 0
        }
        self.available_episodes = list(self.episode_map.keys())

    def __iter__(self) -> Iterator[int]:
        # 1. Deterministic shuffling (epoch-based)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            indices = torch.randperm(len(self.available_episodes), generator=g).tolist()
        else:
            indices = list(range(len(self.available_episodes)))

        # 2. DDP padding (ensure all ranks have equal episode count)
        if not self.drop_last:
            padding_size = self.total_size_episodes - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[: self.total_size_episodes]

        # 3. Subsample for this rank (strided for better load balancing)
        my_episode_indices = indices[self.rank : self.total_size_episodes : self.num_replicas]

        # 4. Flatten into frame indices (episode-contiguous for cache locality)
        final_indices: List[int] = []
        for idx_in_list in my_episode_indices:
            real_ep_id = self.available_episodes[idx_in_list]
            frames = self.episode_map[real_ep_id]
            final_indices.extend(frames)

        return iter(final_indices)

    def __len__(self) -> int:
        total_frames = sum(len(self.episode_map[ep]) for ep in self.available_episodes)
        return math.ceil(total_frames / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic cross-epoch shuffling."""
        self.epoch = epoch
