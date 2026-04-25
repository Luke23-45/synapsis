"""
Data Integration Layer for SYNAPSE Applied Validation
======================================================

Adapted from: applied/robotics/datasets/data.py (source of truth)

Original purpose: NTH-Attention training dataset with torch DataLoaders.
Current purpose:  Pure analytical data loading for empirical validation.
                  No training, no models, no policy code, no torch dependency.

Key data format (from original data.py):
    - proprio_dim:  22  (proprio_0 .. proprio_21)
    - action_dim:    8  (actions_0 .. actions_7)
    - image_size:  224  (not used here — analytical only)
    - num_phases:    5  (gt_phase_0: 0.0 .. 4.0)
    - expert_states: MOVE_TO_PRE_GRASP, DESCEND_TO_GRASP, GRASP, LIFT,
                     MOVE_TO_GOAL, DESCEND_TO_PLACE, PREPARE_PLACE,
                     RELEASE, AWAIT_STABLE_PLACEMENT, RETRACT,
                     PREPARE_GRIPPER, DONE

Data sources:
    - applied/robotics/output_data/exports/*/telemetry.csv
    - applied/robotics/output_data/raw/final_training_set/training_set_index.json
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (source of truth: applied/robotics/datasets/data.py)
# ---------------------------------------------------------------------------

PROPRIO_DIM = 22
ACTION_DIM = 8
EE_POSE_DIM = 7
EE_VEL_DIM = 6
OBJECT_POS_DIM = 3
NUM_PHASES = 5             # gt_phase_0 ranges 0..4

PROPRIO_COLS = [f"proprio_{i}" for i in range(PROPRIO_DIM)]
ACTION_COLS = [f"actions_{i}" for i in range(ACTION_DIM)]
EE_POSE_COLS = [f"ee_pose_world_{i}" for i in range(EE_POSE_DIM)]
EE_VEL_COLS = [f"ee_vel_{i}" for i in range(EE_VEL_DIM)]
OBJECT_POS_COLS = [f"object_pos_world_{i}" for i in range(OBJECT_POS_DIM)]

PHASE_COL = "gt_phase_0"
EXPERT_STATE_COL = "expert_states"
GRASP_COL = "is_grasped_0"
STEP_COL = "step"
EPISODE_ID_COL = "episode_id"

REQUIRED_COLS = (
    [EPISODE_ID_COL, STEP_COL, PHASE_COL, EXPERT_STATE_COL]
    + PROPRIO_COLS
    + ACTION_COLS
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RobotEpisode:
    """One robotics episode loaded from telemetry CSV.

    Mirrors the data shapes from the original NTHDataset / SyntheticNTHDataset:
        proprio:       (T, 22)  float32
        actions:       (T, 8)   float32
        ee_pose:       (T, 7)   float32
        ee_vel:        (T, 6)   float32
        object_pos:    (T, 3)   float32
        gt_phase:      (T,)     int64      (was phase_labels in NTHDataset)
        expert_states: list[str]            (semantic labels)
    """
    episode_id: str
    length: int
    proprio: np.ndarray           # (T, 22)
    actions: np.ndarray           # (T, 8)
    ee_pose: np.ndarray           # (T, 7)
    ee_vel: np.ndarray            # (T, 6)
    object_pos: np.ndarray        # (T, 3)
    expert_states: List[str]      # T labels
    gt_phase: np.ndarray          # (T,) int64
    is_grasped: np.ndarray        # (T,) float32
    success: Optional[bool] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def phase_boundaries(self) -> List[int]:
        """Timestep indices where gt_phase changes."""
        return [t for t in range(1, len(self.gt_phase))
                if self.gt_phase[t] != self.gt_phase[t - 1]]

    @property
    def expert_state_boundaries(self) -> List[int]:
        """Timestep indices where expert_states changes (finer-grained)."""
        return [t for t in range(1, len(self.expert_states))
                if self.expert_states[t] != self.expert_states[t - 1]]

    @property
    def unique_expert_states(self) -> List[str]:
        """Ordered distinct expert_states sequence."""
        seen: List[str] = []
        for s in self.expert_states:
            if not seen or seen[-1] != s:
                seen.append(s)
        return seen

    @property
    def state_sequence(self) -> np.ndarray:
        """Primary state sequence for SYNAPSE memory operator: proprio (T, 22)."""
        return self.proprio


@dataclass
class AppliedDataset:
    """Collection of robot episodes with index metadata."""
    episodes: List[RobotEpisode]
    index_metadata: Optional[Dict] = None
    export_root: str = ""

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def total_steps(self) -> int:
        return sum(ep.length for ep in self.episodes)

    def summary(self) -> Dict:
        lengths = [ep.length for ep in self.episodes]
        return {
            "num_episodes": self.num_episodes,
            "total_steps": self.total_steps,
            "min_length": int(np.min(lengths)) if lengths else 0,
            "max_length": int(np.max(lengths)) if lengths else 0,
            "mean_length": float(np.mean(lengths)) if lengths else 0,
            "successful": sum(1 for ep in self.episodes if ep.success),
        }


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _extract_cols(row: Dict[str, str], cols: List[str]) -> np.ndarray:
    return np.array([_parse_float(row.get(c, "0.0")) for c in cols],
                    dtype=np.float32)


def load_telemetry_episode(csv_path: Path) -> RobotEpisode:
    """Load a single episode from a telemetry.csv file."""
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty telemetry file: {csv_path}")

    available = set(rows[0].keys())
    missing = set(REQUIRED_COLS) - available
    if missing:
        raise ValueError(
            f"Missing columns in {csv_path}: {sorted(missing)}")

    T = len(rows)
    episode_id = rows[0].get(EPISODE_ID_COL, csv_path.parent.name)

    proprio = np.zeros((T, PROPRIO_DIM), dtype=np.float32)
    actions = np.zeros((T, ACTION_DIM), dtype=np.float32)
    ee_pose = np.zeros((T, EE_POSE_DIM), dtype=np.float32)
    ee_vel = np.zeros((T, EE_VEL_DIM), dtype=np.float32)
    object_pos = np.zeros((T, OBJECT_POS_DIM), dtype=np.float32)
    gt_phase = np.zeros(T, dtype=np.int64)
    is_grasped = np.zeros(T, dtype=np.float32)
    expert_states: List[str] = []

    for t, row in enumerate(rows):
        proprio[t] = _extract_cols(row, PROPRIO_COLS)
        actions[t] = _extract_cols(row, ACTION_COLS)
        ee_pose[t] = _extract_cols(row, EE_POSE_COLS)
        ee_vel[t] = _extract_cols(row, EE_VEL_COLS)
        object_pos[t] = _extract_cols(row, OBJECT_POS_COLS)
        gt_phase[t] = int(_parse_float(row.get(PHASE_COL, "0")))
        is_grasped[t] = _parse_float(row.get(GRASP_COL, "0"))
        expert_states.append(row.get(EXPERT_STATE_COL, "UNKNOWN"))

    return RobotEpisode(
        episode_id=episode_id,
        length=T,
        proprio=proprio,
        actions=actions,
        ee_pose=ee_pose,
        ee_vel=ee_vel,
        object_pos=object_pos,
        expert_states=expert_states,
        gt_phase=gt_phase,
        is_grasped=is_grasped,
    )


def load_index_metadata(index_path: Path) -> Dict:
    """Load training_set_index.json."""
    if not index_path.exists():
        log.warning("Index file not found: %s", index_path)
        return {}
    return json.loads(index_path.read_text(encoding="utf-8"))


def load_applied_dataset(
    export_root: str,
    index_json: str = "",
    max_episodes: int = 0,
) -> AppliedDataset:
    """Load all robot episodes from the exports directory.

    Parameters
    ----------
    export_root : str
        Path to ``applied/robotics/output_data/exports/``
    index_json : str
        Path to ``training_set_index.json`` for metadata.
    max_episodes : int
        If > 0, load at most this many (smoke mode).
    """
    root = Path(export_root)
    if not root.exists():
        raise FileNotFoundError(f"Export root not found: {root}")

    episode_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / "telemetry.csv").exists()
    )

    if max_episodes > 0:
        episode_dirs = episode_dirs[:max_episodes]

    log.info("Loading %d episodes from %s", len(episode_dirs), root)

    # Index metadata for success/seed enrichment
    index_meta: Dict[str, Dict] = {}
    if index_json:
        idx = load_index_metadata(Path(index_json))
        for ep_info in idx.get("episodes", []):
            index_meta[ep_info["episode_id"]] = ep_info

    episodes: List[RobotEpisode] = []
    for ep_dir in episode_dirs:
        try:
            episode = load_telemetry_episode(ep_dir / "telemetry.csv")

            # Enrich with index metadata
            for candidate in [episode.episode_id, ep_dir.name]:
                if candidate in index_meta:
                    meta = index_meta[candidate]
                    episode.success = meta.get("success")
                    episode.metadata = meta
                    break

            episodes.append(episode)
            log.debug("  Loaded %s: T=%d, states=%s",
                       episode.episode_id, episode.length,
                       episode.unique_expert_states)
        except Exception as e:
            log.error("Failed to load %s: %s", ep_dir.name, e)

    idx_full = load_index_metadata(Path(index_json)) if index_json else None

    dataset = AppliedDataset(
        episodes=episodes,
        index_metadata=idx_full,
        export_root=str(root),
    )
    log.info("Loaded dataset: %s", dataset.summary())
    return dataset


def load_applied_dataset_from_lmdb(
    lmdb_path: str,
    max_episodes: int = 0,
) -> AppliedDataset:
    """Load an AppliedDataset from an LMDB file (the SOTA SoA format).

    This is the LMDB equivalent of ``load_applied_dataset``. It reads the
    optimized on-disk format via ``ExpertTrajectoryDataset`` and converts
    each episode into a ``RobotEpisode`` dataclass, producing the same
    ``AppliedDataset`` object that the CSV pipeline yields.

    Design:
    - **Structural parity:** Every ``RobotEpisode`` field is populated from
      the corresponding LMDB modality. Missing optional modalities (e.g.,
      ``ee_vel``, ``object_pos``) are filled with zero arrays of the
      correct shape, ensuring downstream code never encounters ``None``
      or missing-key errors.
    - **expert_states → state_names:** The pickled ``expert_states`` list
      from LMDB is mapped to the ``expert_states`` field of RobotEpisode.
    - **Index metadata enrichment:** Episode-level metadata (success, seed)
      from the LMDB JSON index is injected into each RobotEpisode, matching
      the enrichment that ``load_applied_dataset`` performs from the
      training_set_index.json.
    - **Error resilience:** Individual episode conversion failures are
      logged and skipped, mirroring the CSV loader's robustness.

    Parameters
    ----------
    lmdb_path : str
        Path to the ``.lmdb`` file (must have a companion ``_index.json``).
    max_episodes : int
        If > 0, load at most this many episodes (smoke mode).

    Returns
    -------
    AppliedDataset
        With ``episodes`` as ``List[RobotEpisode]`` and ``index_metadata``
        populated from the LMDB index.

    Raises
    ------
    FileNotFoundError
        If the LMDB file does not exist.
    ValueError
        If required modalities are missing from the LMDB index.
    """
    from experiments.empirical.dataset.expert_dataset import StandaloneLMDBReader

    path = Path(lmdb_path)
    if not path.exists():
        raise FileNotFoundError(f"LMDB dataset not found: {path}")

    # Prefer StandaloneLMDBReader (no SYNAPSIS/cv2 dependency).
    # Falls back to ExpertTrajectoryDataset if SYNAPSIS is available and
    # the user needs full SOTA features (LRU cache, image decode, etc.).
    try:
        ds = StandaloneLMDBReader(lmdb_path=str(path))
    except Exception:
        from experiments.empirical.dataset.expert_dataset import ExpertTrajectoryDataset
        log.info("StandaloneLMDBReader unavailable, falling back to ExpertTrajectoryDataset.")
        ds = ExpertTrajectoryDataset(
            demo_path=str(path),
            observation_horizon=2,
            action_horizon=1,
        )

    result = ds.to_applied_dataset(max_episodes=max_episodes)
    raw_episodes = result["episodes"]
    index_metadata = result.get("index_metadata", {})

    # Convert raw dicts → RobotEpisode dataclass instances
    episodes: List[RobotEpisode] = []
    for ep_dict in raw_episodes:
        try:
            T = ep_dict["length"]
            proprio = ep_dict["states"]                          # (T, proprio_dim)
            actions = ep_dict["actions"]                        # (T, action_dim)
            gt_phase = ep_dict["phase_labels"]                  # (T,)

            # expert_states / state_names
            expert_states: List[str] = ep_dict.get("state_names", [])
            if not expert_states:
                expert_states = [f"PHASE_{int(gt_phase[t])}" for t in range(T)]

            # Optional modalities — zero-fill if absent
            ee_pose = ep_dict.get("ee_pose", np.zeros((T, EE_POSE_DIM), dtype=np.float32))
            ee_vel = ep_dict.get("ee_vel", np.zeros((T, EE_VEL_DIM), dtype=np.float32))
            object_pos = ep_dict.get("object_pos", np.zeros((T, OBJECT_POS_DIM), dtype=np.float32))
            is_grasped = ep_dict.get("is_grasped", np.zeros(T, dtype=np.float32))

            episode = RobotEpisode(
                episode_id=ep_dict.get("episode_id", "unknown"),
                length=T,
                proprio=proprio.astype(np.float32),
                actions=actions.astype(np.float32),
                ee_pose=ee_pose.astype(np.float32) if ee_pose is not None else np.zeros((T, EE_POSE_DIM), dtype=np.float32),
                ee_vel=ee_vel.astype(np.float32) if ee_vel is not None else np.zeros((T, EE_VEL_DIM), dtype=np.float32),
                object_pos=object_pos.astype(np.float32) if object_pos is not None else np.zeros((T, OBJECT_POS_DIM), dtype=np.float32),
                expert_states=expert_states,
                gt_phase=gt_phase.astype(np.int64),
                is_grasped=is_grasped.astype(np.float32),
                success=ep_dict.get("success", None),
                metadata={
                    k: ep_dict[k]
                    for k in ("seed", "expert_target_pose", "delta_ee_pose", "gt_gripper")
                    if k in ep_dict
                },
            )
            episodes.append(episode)
        except Exception as e:
            ep_id = ep_dict.get("episode_id", "unknown")
            log.error("Failed to convert LMDB episode %s to RobotEpisode: %s", ep_id, e)

    # Explicitly close the LMDB environment to release the file handle
    ds.close_env()

    dataset = AppliedDataset(
        episodes=episodes,
        index_metadata=index_metadata,
        export_root=str(path),
    )
    log.info("Loaded LMDB dataset: %s", dataset.summary())
    return dataset


# ---------------------------------------------------------------------------
# Synthetic fallback (adapted from SyntheticNTHDataset in original data.py)
# ---------------------------------------------------------------------------

def generate_synthetic_episodes(
    num_episodes: int = 10,
    T: int = 100,
    seed: int = 42,
) -> AppliedDataset:
    """Generate synthetic episodes matching the real data format.

    Used for smoke tests when real data is unavailable.
    Mirrors SyntheticNTHDataset shapes: proprio_dim=22, action_dim=8, num_phases=5.
    """
    rng = np.random.default_rng(seed)
    expert_state_sequence = [
        "MOVE_TO_PRE_GRASP", "DESCEND_TO_GRASP", "GRASP",
        "LIFT", "MOVE_TO_GOAL", "DESCEND_TO_PLACE",
        "PREPARE_PLACE", "RELEASE", "AWAIT_STABLE_PLACEMENT",
        "RETRACT", "PREPARE_GRIPPER", "DONE",
    ]

    episodes: List[RobotEpisode] = []
    for i in range(num_episodes):
        # Phase structure: distribute states across T
        n_states = len(expert_state_sequence)
        segment = max(1, T // n_states)
        states = []
        gt_phase = np.zeros(T, dtype=np.int64)
        for t in range(T):
            state_idx = min(t // segment, n_states - 1)
            states.append(expert_state_sequence[state_idx])
            gt_phase[t] = min(state_idx * NUM_PHASES // n_states, NUM_PHASES - 1)

        episodes.append(RobotEpisode(
            episode_id=f"synthetic_{i:03d}",
            length=T,
            proprio=rng.normal(scale=0.1, size=(T, PROPRIO_DIM)).astype(np.float32),
            actions=rng.normal(scale=0.05, size=(T, ACTION_DIM)).astype(np.float32),
            ee_pose=rng.normal(scale=0.1, size=(T, EE_POSE_DIM)).astype(np.float32),
            ee_vel=rng.normal(scale=0.01, size=(T, EE_VEL_DIM)).astype(np.float32),
            object_pos=rng.normal(scale=0.1, size=(T, OBJECT_POS_DIM)).astype(np.float32),
            expert_states=states,
            gt_phase=gt_phase,
            is_grasped=np.zeros(T, dtype=np.float32),
            success=True,
        ))

    return AppliedDataset(episodes=episodes)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_episode(episode: RobotEpisode) -> List[str]:
    """Check episode integrity. Returns list of issues (empty = valid)."""
    issues: List[str] = []

    if episode.length == 0:
        issues.append("zero-length episode")
        return issues

    if episode.proprio.shape != (episode.length, PROPRIO_DIM):
        issues.append(f"proprio shape {episode.proprio.shape} != ({episode.length}, {PROPRIO_DIM})")

    if episode.actions.shape != (episode.length, ACTION_DIM):
        issues.append(f"actions shape {episode.actions.shape} != ({episode.length}, {ACTION_DIM})")

    if len(episode.expert_states) != episode.length:
        issues.append(f"expert_states len {len(episode.expert_states)} != {episode.length}")

    if len(episode.gt_phase) != episode.length:
        issues.append(f"gt_phase len {len(episode.gt_phase)} != {episode.length}")

    if not np.all(np.isfinite(episode.proprio)):
        issues.append("proprio contains non-finite values")

    if not np.all(np.isfinite(episode.actions)):
        issues.append("actions contains non-finite values")

    phase_vals = np.unique(episode.gt_phase)
    if len(phase_vals) < 2 and episode.length > 20:
        issues.append(f"gt_phase has only {len(phase_vals)} unique value(s)")

    if all(s == "UNKNOWN" for s in episode.expert_states):
        issues.append("all expert_states are UNKNOWN")

    return issues


def validate_dataset(dataset: AppliedDataset) -> Dict:
    """Validate entire dataset. Returns audit dict."""
    audit: Dict = {
        "num_episodes": dataset.num_episodes,
        "total_steps": dataset.total_steps,
        "valid_episodes": 0,
        "invalid_episodes": 0,
        "issues": {},
    }
    for ep in dataset.episodes:
        issues = validate_episode(ep)
        if issues:
            audit["invalid_episodes"] += 1
            audit["issues"][ep.episode_id] = issues
        else:
            audit["valid_episodes"] += 1
    audit["all_valid"] = audit["invalid_episodes"] == 0
    return audit
