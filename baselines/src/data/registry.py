"""
Dataset Registry — Adapter Factory
====================================

Maps dataset names to adapter constructors. Provides a single entry point
for loading any supported dataset by name, abstracting away format differences.

Usage:
    adapter = create_adapter("pusht", local_path=Path("data/external/pusht/train.parquet"))
    episodes = adapter.load_episodes()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from .adapters.base_adapter import BaseDatasetAdapter
from .adapters.lerobot_adapter import LeRobotAdapter, LEROBOT_SPECS

log = logging.getLogger(__name__)


def create_adapter(
    dataset_name: str,
    local_path: Optional[Path] = None,
    max_episodes: Optional[int] = None,
    **kwargs,
) -> BaseDatasetAdapter:
    """Factory function to create the appropriate adapter for a dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset. Must be one of:
        - "pusht", "aloha_transfer", "xarm_lift" (LeRobot HuggingFace)
    local_path : Path or None
        Path to local Parquet file for the dataset.
    max_episodes : int or None
        Limit number of episodes loaded.
    **kwargs
        Additional keyword arguments passed to the adapter constructor.

    Returns
    -------
    BaseDatasetAdapter
    """
    if dataset_name in LEROBOT_SPECS:
        return LeRobotAdapter(
            dataset_name=dataset_name,
            local_path=local_path,
            max_episodes=max_episodes,
        )

    raise ValueError(
        f"Unknown dataset: '{dataset_name}'. "
        f"Available: {list(LEROBOT_SPECS.keys())}"
    )


def list_available_datasets() -> Dict[str, dict]:
    """Return metadata about all available datasets.

    Returns
    -------
    dict mapping dataset_name → metadata dict
    """
    datasets = {}

    for name, spec in LEROBOT_SPECS.items():
        datasets[name] = {
            "source": "lerobot",
            "hf_repo": spec.hf_repo,
            "proprio_dim": spec.proprio_dim,
            "action_dim": spec.action_dim,
            "num_phases": spec.num_phases,
            "max_episode_length": spec.max_episode_length,
        }

    return datasets
