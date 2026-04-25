"""
Dataset adapters for SYNAPSE empirical experiments.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from .tasks import generate_control_dataset

log = logging.getLogger(__name__)


def load_public_state_dataset(path: str, smoke_fallback: bool, seed: int) -> List[Dict[str, np.ndarray]]:
    if not path:
        if smoke_fallback:
            rng = np.random.default_rng(seed)
            return generate_control_dataset(rng, n_episodes=10, T=80, state_dim=12, action_dim=4)
        raise FileNotFoundError("public_benchmark.dataset_path is not set")

    dataset_path = Path(path)
    if not dataset_path.exists():
        if smoke_fallback:
            rng = np.random.default_rng(seed)
            return generate_control_dataset(rng, n_episodes=10, T=80, state_dim=12, action_dim=4)
        raise FileNotFoundError(f"Public benchmark path does not exist: {dataset_path}")

    if dataset_path.suffix == ".npz":
        data = np.load(dataset_path)
        return [{"states": data["states"], "actions": data["actions"]}]

    if dataset_path.is_dir():
        episodes: List[Dict[str, np.ndarray]] = []
        for episode_file in sorted(dataset_path.glob("*.npz")):
            data = np.load(episode_file)
            episodes.append({"states": data["states"], "actions": data["actions"]})
        if episodes:
            return episodes

    raise ValueError(f"Unsupported public benchmark format at {dataset_path}")


def load_inhouse_export_episodes(export_root: str) -> List[Dict[str, np.ndarray]]:
    root = Path(export_root)
    if not root.exists():
        raise FileNotFoundError(f"In-house export root not found: {root}")
    episodes: List[Dict[str, np.ndarray]] = []
    for episode_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        telemetry = episode_dir / "telemetry.csv"
        if not telemetry.exists():
            continue
        with telemetry.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            continue
        actions = np.asarray([[float(row[f"actions_{i}"]) for i in range(8)] for row in rows], dtype=np.float32)
        proprio = np.asarray([[float(row[f"proprio_{i}"]) for i in range(22)] for row in rows], dtype=np.float32)
        phase = np.asarray([int(float(row["gt_phase_0"])) for row in rows], dtype=np.int64)
        episodes.append(
            {
                "episode_id": episode_dir.name,
                "states": proprio,
                "actions": actions,
                "phase_labels": phase,
                "state_names": [row.get("expert_states", "UNKNOWN") for row in rows],
            }
        )
    return episodes


def load_inhouse_index_metadata(index_json: str) -> dict:
    path = Path(index_json)
    if not path.exists():
        raise FileNotFoundError(f"In-house index json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_episodes_from_lmdb(lmdb_path: str, max_episodes: int = 0) -> List[Dict[str, np.ndarray]]:
    """Load episodes from an LMDB dataset file (the SOTA SoA format).

    This is the LMDB equivalent of ``load_inhouse_export_episodes``.
    It uses ``ExpertTrajectoryDataset`` to read the optimized on-disk
    format and materializes episodes into the same plain-dict structure
    that the experiment pipeline expects.

    The returned list schema matches ``load_inhouse_export_episodes``:
        [
            {
                "episode_id":   str,
                "states":       np.ndarray (T, proprio_dim),  float32
                "actions":      np.ndarray (T, action_dim),   float32
                "phase_labels": np.ndarray (T,),              int64
                "state_names":  List[str],
                ...             (optional modalities if present)
            },
            ...
        ]

    Args:
        lmdb_path: Path to the ``.lmdb`` file.
        max_episodes: If > 0, load at most this many episodes (smoke mode).

    Returns:
        List of episode dicts.

    Raises:
        FileNotFoundError: If the LMDB file does not exist.
        ValueError: If the LMDB file lacks a companion index JSON.
    """
    from experiments.empirical.dataset.expert_dataset import ExpertTrajectoryDataset

    path = Path(lmdb_path)
    if not path.exists():
        raise FileNotFoundError(f"LMDB dataset not found: {path}")
    if path.suffix != ".lmdb":
        raise ValueError(f"Expected .lmdb file, got: {path}")

    # ExpertTrajectoryDataset requires observation_horizon and action_horizon,
    # but for analytical materialization we only need minimal values.
    # These horizons affect virtual chunk indexing (used by __getitem__),
    # NOT the to_robot_episode / to_applied_dataset methods which load
    # full episodes directly.
    ds = ExpertTrajectoryDataset(
        demo_path=str(path),
        observation_horizon=2,
        action_horizon=1,
    )

    result = ds.to_applied_dataset(max_episodes=max_episodes)
    episodes = result["episodes"]

    log.info(
        "Loaded %d episodes from LMDB (%s)",
        len(episodes), path.name,
    )

    # Explicitly close the LMDB environment to release the file handle
    ds.close_env()

    return episodes

