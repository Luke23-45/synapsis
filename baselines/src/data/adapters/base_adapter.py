"""
Data Adapters — Base Interface
==============================

Abstract interface that all dataset adapters must implement.
This ensures a uniform RobotEpisode output regardless of the source format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RobotEpisode:
    """Unified episode representation consumed by all downstream components.

    Every adapter must produce episodes in this exact format.
    Fields with ``None`` are optional enrichments that may not be available
    for all datasets (e.g., HuggingFace datasets lack ``ee_pose``).
    """

    episode_id: str
    dataset_name: str
    proprio_history: np.ndarray          # (T, D_proprio)
    actions: np.ndarray                  # (T, D_action)
    gt_phase: np.ndarray                 # (T,) int64

    # Optional enrichments (available for D1 pick-and-place, not for HF datasets)
    ee_pose_history: Optional[np.ndarray] = None    # (T, 7)
    ee_vel_history: Optional[np.ndarray] = None     # (T, 6)
    object_pos_history: Optional[np.ndarray] = None # (T, 3)
    is_grasped_history: Optional[np.ndarray] = None # (T, 1)

    # SYNAPSE features (populated by cache pipeline)
    synapse_anchors: Optional[np.ndarray] = None
    synapse_topo: Optional[np.ndarray] = None

    # Reward signal (if available from LeRobot datasets)
    rewards: Optional[np.ndarray] = None            # (T,)
    success: bool = False

    @property
    def length(self) -> int:
        return int(self.proprio_history.shape[0])

    @property
    def structured_history(self) -> np.ndarray:
        """Build the full structured state by concatenating all available modalities.

        Falls back to proprio_history if no enrichments are available.
        This ensures M() always receives the richest available signal.
        """
        parts = [self.proprio_history]
        if self.ee_pose_history is not None:
            parts.append(self.ee_pose_history)
        if self.ee_vel_history is not None:
            parts.append(self.ee_vel_history)
        if self.object_pos_history is not None:
            parts.append(self.object_pos_history)
        if self.is_grasped_history is not None:
            parts.append(self.is_grasped_history)
        return np.concatenate(parts, axis=1).astype(np.float32)

    @property
    def structured_state_dim(self) -> int:
        return self.structured_history.shape[1]

    def apply_structured_normalization(self, normalized: np.ndarray) -> None:
        """Apply pre-computed Z-score normalization back to individual modalities.

        This mirrors the original implementation but handles the case where
        some modalities may be absent (HuggingFace datasets).
        """
        offset = 0
        p = self.proprio_history.shape[1]
        self.proprio_history = normalized[:, offset:offset + p].astype(np.float32)
        offset += p

        if self.ee_pose_history is not None:
            d = self.ee_pose_history.shape[1]
            self.ee_pose_history = normalized[:, offset:offset + d].astype(np.float32)
            offset += d

        if self.ee_vel_history is not None:
            d = self.ee_vel_history.shape[1]
            self.ee_vel_history = normalized[:, offset:offset + d].astype(np.float32)
            offset += d

        if self.object_pos_history is not None:
            d = self.object_pos_history.shape[1]
            self.object_pos_history = normalized[:, offset:offset + d].astype(np.float32)
            offset += d

        if self.is_grasped_history is not None:
            d = self.is_grasped_history.shape[1]
            self.is_grasped_history = normalized[:, offset:offset + d].astype(np.float32)
            offset += d


class BaseDatasetAdapter(ABC):
    """Abstract base for all dataset adapters.

    Each adapter reads a specific data format (e.g., LeRobot Parquet)
    and produces a list of :class:`RobotEpisode` instances in a consistent
    format suitable for the SYNAPSE baselines experiment.
    """

    @abstractmethod
    def load_episodes(self) -> List[RobotEpisode]:
        """Load all episodes from the source and return as RobotEpisode list.

        Returns
        -------
        list of RobotEpisode
            Episodes in the unified format.
        """
        ...

    @property
    @abstractmethod
    def proprio_dim(self) -> int:
        """Dimensionality of the proprioceptive state vector."""
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the action vector."""
        ...

    @property
    @abstractmethod
    def structured_state_dim(self) -> int:
        """Total dimensionality of the structured state (proprio + enrichments)."""
        ...

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Canonical name for this dataset (used in configs and reports)."""
        ...

    @property
    @abstractmethod
    def num_phases(self) -> int:
        """Number of discrete task phases for phase classification."""
        ...
