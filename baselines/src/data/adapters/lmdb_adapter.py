"""
LMDB Adapter - Pick-and-Place Dataset Integration
=================================================

Loads the robotics LMDB dataset used by the end-to-end SYNAPSE pipeline and
converts it into the baseline ``RobotEpisode`` format.

This adapter preserves the richer structured state available in the LMDB
export: proprio, end-effector pose/velocity, object position, and grasp flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base_adapter import BaseDatasetAdapter, RobotEpisode

log = logging.getLogger(__name__)

try:
    from experiments.end_to_end.data.e2e_expert_dataset import StandaloneLMDBReader
except Exception:  # pragma: no cover - exercised in integration, not unit tests
    StandaloneLMDBReader = None


@dataclass(frozen=True)
class LMDBDatasetSpec:
    name: str
    proprio_dim: int
    action_dim: int
    num_phases: int
    ee_pose_dim: int
    ee_vel_dim: int
    object_pos_dim: int
    grasp_dim: int


LMDB_SPECS: Dict[str, LMDBDatasetSpec] = {
    "pick_place": LMDBDatasetSpec(
        name="pick_place",
        proprio_dim=22,
        action_dim=8,
        num_phases=5,
        ee_pose_dim=7,
        ee_vel_dim=6,
        object_pos_dim=3,
        grasp_dim=1,
    ),
}


class LMDBAdapter(BaseDatasetAdapter):
    """Adapter for the robotics LMDB dataset used by end-to-end training."""

    def __init__(
        self,
        dataset_name: str,
        local_path: Path,
        max_episodes: Optional[int] = None,
    ) -> None:
        if dataset_name not in LMDB_SPECS:
            raise ValueError(
                f"Unknown LMDB dataset: {dataset_name}. "
                f"Available: {list(LMDB_SPECS.keys())}"
            )
        self._spec = LMDB_SPECS[dataset_name]
        self._local_path = Path(local_path)
        self._max_episodes = max_episodes

    @property
    def proprio_dim(self) -> int:
        return self._spec.proprio_dim

    @property
    def action_dim(self) -> int:
        return self._spec.action_dim

    @property
    def structured_state_dim(self) -> int:
        return (
            self._spec.proprio_dim
            + self._spec.ee_pose_dim
            + self._spec.ee_vel_dim
            + self._spec.object_pos_dim
            + self._spec.grasp_dim
        )

    @property
    def dataset_name(self) -> str:
        return self._spec.name

    @property
    def num_phases(self) -> int:
        return self._spec.num_phases

    def load_episodes(self) -> List[RobotEpisode]:
        if not self._local_path.exists():
            raise FileNotFoundError(f"LMDB dataset not found: {self._local_path}")
        if StandaloneLMDBReader is None:
            raise ImportError(
                "StandaloneLMDBReader is unavailable. Ensure end-to-end LMDB "
                "dependencies are installed."
            )

        reader = StandaloneLMDBReader(str(self._local_path))
        try:
            result = reader.to_applied_dataset(max_episodes=self._max_episodes or 0)
        finally:
            reader.close_env()

        episodes = [self._convert_episode(ep) for ep in result["episodes"]]
        log.info(
            "LMDBAdapter[%s]: loaded %d episodes from %s",
            self._spec.name,
            len(episodes),
            self._local_path,
        )
        return episodes

    def _convert_episode(self, ep: dict) -> RobotEpisode:
        T = int(ep["length"])

        ee_pose = self._optional_matrix(ep.get("ee_pose"), T, self._spec.ee_pose_dim)
        ee_vel = self._optional_matrix(ep.get("ee_vel"), T, self._spec.ee_vel_dim)
        object_pos = self._optional_matrix(ep.get("object_pos"), T, self._spec.object_pos_dim)
        is_grasped = self._optional_vector(ep.get("is_grasped"), T)

        return RobotEpisode(
            episode_id=str(ep.get("episode_id", "unknown")),
            dataset_name=self._spec.name,
            proprio_history=np.asarray(ep["states"], dtype=np.float32),
            actions=np.asarray(ep["actions"], dtype=np.float32),
            gt_phase=np.asarray(ep["phase_labels"], dtype=np.int64).reshape(T),
            ee_pose_history=ee_pose,
            ee_vel_history=ee_vel,
            object_pos_history=object_pos,
            is_grasped_history=is_grasped,
            success=bool(ep.get("success", False)),
        )

    @staticmethod
    def _optional_matrix(value: Optional[np.ndarray], T: int, D: int) -> np.ndarray:
        if value is None:
            return np.zeros((T, D), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        return arr.reshape(T, D)

    @staticmethod
    def _optional_vector(value: Optional[np.ndarray], T: int) -> np.ndarray:
        if value is None:
            return np.zeros((T, 1), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32).reshape(T, -1)
        if arr.shape[1] == 1:
            return arr
        return arr[:, :1]
