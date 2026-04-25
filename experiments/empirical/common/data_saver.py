"""
experiments/empirical/common/data_saver.py

Utility functions allowing individual experiment scripts to handle their own
data saving for raw publication artifacts (e.g., JSONL, NPZ).
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

log = logging.getLogger(__name__)


def save_experiment_npz(
    experiment_id: str,
    seed: int,
    data: Dict[str, Any],
    output_dir: str = "experiments/outputs/empirical/raw_data"
) -> Path:
    """
    Save dense multi-dimensional matrices and trajectories to a highly compressed
    Numpy archive (.npz).
    
    This is vastly superior to JSON/JSONL for raw trajectories (10x smaller, faster load).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = out_dir / f"{experiment_id}_seed{seed}_artifacts.npz"
    
    # Filter and convert everything to numpy arrays to ensure safety
    clean_data = {}
    for k, v in data.items():
        if v is None:
            continue
        clean_data[k] = np.asarray(v)
        
    np.savez_compressed(file_path, **clean_data)
    log.info("[%s] Saved raw NumPy artifacts to %s", experiment_id, file_path)
    return file_path


def save_experiment_jsonl(
    experiment_id: str,
    seed: int,
    records: List[Dict[str, Any]],
    output_dir: str = "experiments/outputs/empirical/raw_data"
) -> Path:
    """
    Save list of structured dictionaries to JSON Lines (.jsonl).
    
    Best used for mixed-type datasets, text labels, or metadata traces.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = out_dir / f"{experiment_id}_seed{seed}_artifacts.jsonl"
    
    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
            
    log.info("[%s] Saved raw JSONL artifacts to %s", experiment_id, file_path)
    return file_path


def save_experiment_json(
    experiment_id: str,
    seed: int,
    data: Dict[str, Any],
    output_dir: str = "experiments/outputs/empirical/raw_data"
) -> Path:
    """
    Save hierarchical structured data to standard JSON.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = out_dir / f"{experiment_id}_seed{seed}_artifacts.json"
    
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
        
    log.info("[%s] Saved raw JSON artifacts to %s", experiment_id, file_path)
    return file_path
