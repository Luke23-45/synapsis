"""
LeRobot Adapter — HuggingFace Dataset Integration
===================================================

Reads datasets in the LeRobot/Parquet format from HuggingFace Hub and
converts them into the unified RobotEpisode representation.

Supports:
  - lerobot/pusht (2D pushing, state_dim=2, action_dim=2)
  - lerobot/aloha_sim_transfer_cube_human (bimanual, state=14, action=14)
  - lerobot/xarm_lift_medium (single-arm, state=4, action=3)

Phase labels are synthesized from episode progress or reward signals
since LeRobot datasets do not provide expert phase annotations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_adapter import BaseDatasetAdapter, RobotEpisode

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeRobotDatasetSpec:
    """Specification for a LeRobot dataset."""

    hf_repo: str
    name: str
    proprio_dim: int
    action_dim: int
    num_phases: int
    max_episode_length: int
    fps: int
    state_key: str = "observation.state"
    action_key: str = "action"
    reward_key: str = "next.reward"
    done_key: str = "next.done"
    success_key: str = "next.success"
    episode_index_key: str = "episode_index"
    frame_index_key: str = "frame_index"
    phase_strategy: str = "progress_ratio"
    reward_thresholds: Tuple[float, ...] = ()


# Pre-defined specifications for all supported LeRobot datasets
LEROBOT_SPECS: Dict[str, LeRobotDatasetSpec] = {
    "pusht": LeRobotDatasetSpec(
        hf_repo="lerobot/pusht",
        name="pusht",
        proprio_dim=2,
        action_dim=2,
        num_phases=3,
        max_episode_length=300,
        fps=10,
        phase_strategy="reward_threshold",
        reward_thresholds=(0.3, 0.7),
    ),
    "aloha_transfer": LeRobotDatasetSpec(
        hf_repo="lerobot/aloha_sim_transfer_cube_human",
        name="aloha_transfer",
        proprio_dim=14,
        action_dim=14,
        num_phases=4,
        max_episode_length=400,
        fps=50,
        phase_strategy="progress_ratio",
    ),
    "xarm_lift": LeRobotDatasetSpec(
        hf_repo="lerobot/xarm_lift_medium",
        name="xarm_lift",
        proprio_dim=4,
        action_dim=3,
        num_phases=3,
        max_episode_length=250,
        fps=15,
        phase_strategy="reward_threshold",
        reward_thresholds=(0.5,),
    ),
}


class LeRobotAdapter(BaseDatasetAdapter):
    """Adapter for HuggingFace LeRobot datasets.

    Loads data from either:
    1. A local Parquet file (downloaded by download_datasets.py)
    2. Directly from HuggingFace Hub via the ``datasets`` library

    Parameters
    ----------
    dataset_name : str
        Key into LEROBOT_SPECS (e.g., "pusht", "aloha_transfer", "xarm_lift").
    local_path : Path or None
        Path to a local Parquet file. If None, downloads from HuggingFace.
    max_episodes : int or None
        Limit the number of episodes loaded (for quick testing).
    """

    def __init__(
        self,
        dataset_name: str,
        local_path: Optional[Path] = None,
        max_episodes: Optional[int] = None,
    ) -> None:
        if dataset_name not in LEROBOT_SPECS:
            raise ValueError(
                f"Unknown LeRobot dataset: {dataset_name}. "
                f"Available: {list(LEROBOT_SPECS.keys())}"
            )

        self._spec = LEROBOT_SPECS[dataset_name]
        self._local_path = Path(local_path) if local_path else None
        self._max_episodes = max_episodes

    @property
    def proprio_dim(self) -> int:
        return self._spec.proprio_dim

    @property
    def action_dim(self) -> int:
        return self._spec.action_dim

    @property
    def structured_state_dim(self) -> int:
        # LeRobot datasets only have state — no separate ee_pose, ee_vel, etc.
        return self._spec.proprio_dim

    @property
    def dataset_name(self) -> str:
        return self._spec.name

    @property
    def num_phases(self) -> int:
        return self._spec.num_phases

    def load_episodes(self) -> List[RobotEpisode]:
        """Load all episodes from the LeRobot dataset.

        Returns
        -------
        list of RobotEpisode
        """
        raw_data = self._load_raw_data()
        episodes = self._convert_to_episodes(raw_data)

        if self._max_episodes is not None:
            episodes = episodes[: self._max_episodes]

        log.info(
            "LeRobotAdapter[%s]: loaded %d episodes "
            "(proprio_dim=%d, action_dim=%d, phases=%d)",
            self._spec.name,
            len(episodes),
            self._spec.proprio_dim,
            self._spec.action_dim,
            self._spec.num_phases,
        )
        return episodes

    def _load_raw_data(self) -> dict:
        """Load raw data from Parquet file or HuggingFace Hub.

        Returns
        -------
        dict mapping column names to numpy arrays
        """
        if self._local_path is not None and self._local_path.exists():
            parquet_path = self._local_path
            if parquet_path.is_dir():
                parquet_path = parquet_path / "train.parquet"
            return self._load_from_parquet(parquet_path)
        else:
            return self._load_from_hub()

    def _load_from_parquet(self, path: Path) -> dict:
        """Load data from a local Parquet file."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet loading. "
                "Install with: pip install pyarrow"
            )

        log.info("Loading from local Parquet: %s", path)
        table = pq.read_table(str(path), memory_map=True)

        result = {}
        for col_name in table.column_names:
            col_data = table.column(col_name)
            try:
                result[col_name] = col_data.to_numpy()
            except Exception:
                # Some columns (like video paths) may not convert to numpy
                result[col_name] = col_data.to_pylist()

        log.info("Loaded %d rows from Parquet", table.num_rows)
        return result

    def _load_from_hub(self) -> dict:
        """Load data directly from HuggingFace Hub."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required for HuggingFace Hub loading. "
                "Install with: pip install datasets"
            )

        log.info("Loading from HuggingFace Hub: %s", self._spec.hf_repo)
        ds = load_dataset(self._spec.hf_repo, split="train")

        result = {}
        for col_name in ds.column_names:
            try:
                result[col_name] = np.array(ds[col_name])
            except Exception:
                result[col_name] = ds[col_name]

        log.info("Loaded %d rows from HuggingFace Hub", len(ds))
        return result

    def _convert_to_episodes(self, raw_data: dict) -> List[RobotEpisode]:
        """Convert flat row-based data into episode-based RobotEpisode list.

        LeRobot data is stored as flat rows with ``episode_index`` identifying
        which episode each row belongs to. We group by episode and extract
        state/action sequences.
        """
        spec = self._spec
        episode_indices = np.asarray(raw_data[spec.episode_index_key]).flatten()
        unique_episodes = np.unique(episode_indices)
        frame_indices = None
        if spec.frame_index_key in raw_data:
            frame_indices = np.asarray(raw_data[spec.frame_index_key]).flatten()

        # Extract state and action arrays
        state_data = self._extract_array(raw_data, spec.state_key, spec.proprio_dim)
        action_data = self._extract_array(raw_data, spec.action_key, spec.action_dim)

        # Extract reward if available
        reward_data = None
        if spec.reward_key in raw_data:
            try:
                reward_data = np.asarray(raw_data[spec.reward_key], dtype=np.float32).flatten()
            except Exception:
                reward_data = None

        # Extract success if available
        success_data = None
        if spec.success_key in raw_data:
            try:
                success_data = np.asarray(raw_data[spec.success_key]).flatten()
            except Exception:
                success_data = None

        episodes: List[RobotEpisode] = []

        for ep_idx in unique_episodes:
            mask = episode_indices == ep_idx
            row_indices = np.flatnonzero(mask)
            if frame_indices is not None and row_indices.size > 1:
                order = np.argsort(frame_indices[row_indices], kind="stable")
                row_indices = row_indices[order]

            ep_state = state_data[row_indices].astype(np.float32, copy=False)
            ep_action = action_data[row_indices].astype(np.float32, copy=False)
            ep_len = ep_state.shape[0]

            # Truncate to max episode length
            if ep_len > spec.max_episode_length:
                ep_state = ep_state[: spec.max_episode_length]
                ep_action = ep_action[: spec.max_episode_length]
                ep_len = spec.max_episode_length

            # Ensure action and state arrays are the same length
            min_len = min(ep_state.shape[0], ep_action.shape[0])
            ep_state = ep_state[:min_len]
            ep_action = ep_action[:min_len]
            ep_len = min_len

            if ep_len < 2:
                log.warning("Skipping episode %d: too short (%d steps)", ep_idx, ep_len)
                continue

            # Extract rewards for this episode
            ep_reward = None
            if reward_data is not None:
                ep_reward = reward_data[row_indices][:ep_len].astype(np.float32, copy=False)

            # Compute phase labels
            gt_phase = self._compute_phase_labels(
                ep_state, ep_reward, ep_len
            )

            # Check success
            ep_success = False
            if success_data is not None:
                ep_success_arr = success_data[row_indices][:ep_len]
                ep_success = bool(np.any(ep_success_arr))

            episodes.append(
                RobotEpisode(
                    episode_id=f"{spec.name}_ep{int(ep_idx):04d}",
                    dataset_name=spec.name,
                    proprio_history=ep_state,
                    actions=ep_action,
                    gt_phase=gt_phase,
                    rewards=ep_reward,
                    success=ep_success,
                )
            )

        return episodes

    def _extract_array(
        self, raw_data: dict, key: str, expected_dim: int
    ) -> np.ndarray:
        """Extract a named array from the raw data dict.

        Handles both flat arrays and nested list-of-lists formats.
        """
        if key not in raw_data:
            raise KeyError(
                f"Required key '{key}' not found in dataset. "
                f"Available keys: {list(raw_data.keys())}"
            )

        data = raw_data[key]

        if isinstance(data, np.ndarray):
            arr = data
        elif isinstance(data, list):
            # May be list of lists or list of scalars
            arr = np.array(data, dtype=np.float32)
        else:
            arr = np.asarray(data, dtype=np.float32)

        # Ensure 2D: (N, D)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if arr.shape[1] != expected_dim:
            log.warning(
                "Dimension mismatch for '%s': expected %d, got %d. "
                "Truncating/padding to match.",
                key, expected_dim, arr.shape[1],
            )
            if arr.shape[1] > expected_dim:
                arr = arr[:, :expected_dim]
            else:
                pad_width = expected_dim - arr.shape[1]
                arr = np.pad(arr, ((0, 0), (0, pad_width)), mode="constant")

        return arr

    def _compute_phase_labels(
        self,
        state: np.ndarray,
        reward: Optional[np.ndarray],
        ep_len: int,
    ) -> np.ndarray:
        """Synthesize task phase labels for this episode.

        Parameters
        ----------
        state : np.ndarray, shape (T, D)
        reward : np.ndarray or None, shape (T,)
        ep_len : int

        Returns
        -------
        np.ndarray, shape (T,), dtype int64
        """
        spec = self._spec

        if spec.phase_strategy == "reward_threshold" and reward is not None:
            return self._phase_from_reward(reward, spec.reward_thresholds, spec.num_phases)
        elif spec.phase_strategy == "progress_ratio":
            return self._phase_from_progress(ep_len, spec.num_phases)
        else:
            # Fallback: uniform progress-based phases
            return self._phase_from_progress(ep_len, spec.num_phases)

    @staticmethod
    def _phase_from_reward(
        reward: np.ndarray,
        thresholds: Tuple[float, ...],
        num_phases: int,
    ) -> np.ndarray:
        """Assign phases based on cumulative reward crossing thresholds.

        For PushT: reward < 0.3 → phase 0 (approach),
                   0.3 ≤ reward < 0.7 → phase 1 (contact),
                   reward ≥ 0.7 → phase 2 (push-to-goal)
        """
        phases = np.zeros(len(reward), dtype=np.int64)
        sorted_thresholds = sorted(thresholds)

        for i, thresh in enumerate(sorted_thresholds):
            phases[reward >= thresh] = i + 1

        # Clamp to valid range
        phases = np.clip(phases, 0, num_phases - 1)
        return phases

    @staticmethod
    def _phase_from_progress(ep_len: int, num_phases: int) -> np.ndarray:
        """Assign phases based on episode progress ratio.

        Divides the episode into num_phases equal segments.
        """
        phase_boundaries = np.linspace(0, ep_len, num_phases + 1, dtype=np.int64)
        phases = np.zeros(ep_len, dtype=np.int64)

        for p in range(num_phases):
            start = int(phase_boundaries[p])
            end = int(phase_boundaries[p + 1])
            phases[start:end] = p

        return phases
