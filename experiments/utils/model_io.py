"""
Model I/O utilities for SYNAPSE experiment artifacts.

Centralises experiment result saving, loading, pointer resolution, and
run capsule management — adapted from GibbsQ's model_io.py pattern.

Usage
-----
::

    from experiments.utils.model_io import (
        create_run_capsule,
        save_experiment_artifact,
        save_run_pointer,
        resolve_latest_run,
    )

    capsule = create_run_capsule("experiments/outputs", "causality")
    save_experiment_artifact(capsule, "anchors", data)
    save_run_pointer(capsule, "experiments/outputs")
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "create_run_capsule",
    "save_experiment_artifact",
    "load_experiment_artifact",
    "save_run_pointer",
    "resolve_latest_run",
    "save_config_snapshot",
    "RunCapsule",
]


class RunCapsule:
    """
    Structured output directory for a single experiment run.

    Layout
    ------
    ::

        outputs/<experiment_type>/<run_id>/
            figures/     — PNG, PDF, SVG charts
            metrics/     — JSONL numeric data
            metadata/    — config snapshot, run info
            artifacts/   — serialised numpy arrays, persistence diagrams
            logs/        — experiment log text
    """

    def __init__(self, base_path: Path):
        self.root = base_path
        self.figures = base_path / "figures"
        self.metrics = base_path / "metrics"
        self.metadata = base_path / "metadata"
        self.artifacts = base_path / "artifacts"
        self.logs = base_path / "logs"

    def create(self) -> "RunCapsule":
        """Create all subdirectories."""
        for d in [self.figures, self.metrics, self.metadata, self.artifacts, self.logs]:
            d.mkdir(parents=True, exist_ok=True)
        return self


def create_run_capsule(
    output_root: str | Path,
    experiment_type: str,
    run_id: Optional[str] = None,
) -> RunCapsule:
    """
    Create a timestamped run capsule directory.

    Parameters
    ----------
    output_root : str or Path
        Base output directory (e.g., "outputs").
    experiment_type : str
        Experiment identifier (e.g., "exp01_causality").
    run_id : str, optional
        Custom run ID. Default: timestamp-based.

    Returns
    -------
    RunCapsule
        Initialized capsule with all subdirectories created.
    """
    output_root = Path(output_root)
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    capsule_path = output_root / experiment_type / run_id
    capsule = RunCapsule(capsule_path)
    capsule.create()

    log.info(f"[RunCapsule] Created: {capsule_path}")
    return capsule


def save_experiment_artifact(
    capsule: RunCapsule,
    name: str,
    data: Any,
    format: str = "npz",
) -> Path:
    """
    Save an experiment artifact (numpy arrays, persistence diagrams, etc.).

    Parameters
    ----------
    capsule : RunCapsule
    name : str
        Artifact name (without extension).
    data : dict or np.ndarray
        Data to save.
    format : str
        'npz', 'json', or 'npy'.

    Returns
    -------
    Path
        Path to saved artifact.
    """
    format = format.lower()

    if format == "npz":
        path = capsule.artifacts / f"{name}.npz"
        if isinstance(data, dict):
            np_data = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    np_data[key] = value
                elif isinstance(value, (list, tuple)):
                    np_data[key] = np.array(value)
                else:
                    np_data[key] = np.array(value)
            np.savez_compressed(path, **np_data)
        elif isinstance(data, np.ndarray):
            np.savez_compressed(path, data=data)
        else:
            raise TypeError(f"Cannot save {type(data)} as npz")

    elif format == "json":
        path = capsule.artifacts / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_serializer)

    elif format == "npy":
        path = capsule.artifacts / f"{name}.npy"
        np.save(path, data)

    else:
        raise ValueError(f"Unsupported format: {format}")

    log.debug(f"[Artifact] Saved: {path}")
    return path


def load_experiment_artifact(
    capsule: RunCapsule,
    name: str,
    format: str = "npz",
) -> Any:
    """
    Load an experiment artifact.

    Parameters
    ----------
    capsule : RunCapsule
    name : str
        Artifact name (without extension).
    format : str
        'npz', 'json', or 'npy'.

    Returns
    -------
    Loaded data.
    """
    format = format.lower()

    if format == "npz":
        path = capsule.artifacts / f"{name}.npz"
        return dict(np.load(path, allow_pickle=False))

    elif format == "json":
        path = capsule.artifacts / f"{name}.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    elif format == "npy":
        path = capsule.artifacts / f"{name}.npy"
        return np.load(path, allow_pickle=False)

    else:
        raise ValueError(f"Unsupported format: {format}")


def save_metrics(
    capsule: RunCapsule,
    metrics: Dict[str, Any],
    filename: str = "metrics.jsonl",
) -> Path:
    """
    Append a metrics record to a JSONL file (one JSON object per line).

    Parameters
    ----------
    capsule : RunCapsule
    metrics : dict
        Key-value pairs to record.
    filename : str
        JSONL filename.

    Returns
    -------
    Path
    """
    path = capsule.metrics / filename
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, default=_json_serializer) + "\n")
    return path


def save_config_snapshot(
    capsule: RunCapsule,
    config: Any,
) -> Path:
    """
    Save a snapshot of the config used for this run.

    Parameters
    ----------
    capsule : RunCapsule
    config : ExperimentConfig or dict
        Configuration to snapshot.

    Returns
    -------
    Path
    """
    path = capsule.metadata / "config.yaml"

    if hasattr(config, "__dataclass_fields__"):
        # Convert dataclass to dict recursively
        import dataclasses
        data = dataclasses.asdict(config)
    elif isinstance(config, dict):
        data = config
    else:
        data = {"config": str(config)}

    try:
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_serializer)

    log.debug(f"[Config] Snapshot saved: {path}")
    return path


def save_run_pointer(
    capsule: RunCapsule,
    output_root: str | Path,
    pointer_name: str = "latest_run.txt",
) -> Path:
    """
    Write a pointer file that records the latest run directory.

    Mirrors GibbsQ's model pointer pattern.

    Parameters
    ----------
    capsule : RunCapsule
    output_root : str or Path
    pointer_name : str

    Returns
    -------
    Path
    """
    output_root = Path(output_root)
    experiment_dir = capsule.root.parent
    ptr_path = experiment_dir / pointer_name

    # Store relative path from experiment_dir
    try:
        relative_path = capsule.root.relative_to(experiment_dir)
    except ValueError:
        relative_path = capsule.root

    ptr_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ptr_path, "w", encoding="utf-8") as f:
        f.write(str(relative_path))

    log.info(f"[Pointer] Updated {pointer_name} → {relative_path}")
    return ptr_path


def resolve_latest_run(
    output_root: str | Path,
    experiment_type: str,
    pointer_name: str = "latest_run.txt",
) -> Optional[RunCapsule]:
    """
    Resolve the latest run capsule from a pointer file.

    Parameters
    ----------
    output_root : str or Path
    experiment_type : str
    pointer_name : str

    Returns
    -------
    RunCapsule or None
    """
    output_root = Path(output_root)
    experiment_dir = output_root / experiment_type
    ptr_path = experiment_dir / pointer_name

    if not ptr_path.exists():
        log.warning(f"No pointer file: {ptr_path}")
        return None

    relative = ptr_path.read_text(encoding="utf-8").strip()
    run_dir = experiment_dir / relative

    if not run_dir.exists():
        log.warning(f"Pointer target does not exist: {run_dir}")
        return None

    return RunCapsule(run_dir)


def find_all_runs(
    output_root: str | Path,
    experiment_type: Optional[str] = None,
) -> List[RunCapsule]:
    """
    Discover all run capsules under an output root.

    Parameters
    ----------
    output_root : str or Path
    experiment_type : str, optional
        Filter to a specific experiment type.

    Returns
    -------
    list of RunCapsule
    """
    output_root = Path(output_root)
    capsules = []

    if experiment_type:
        search_dirs = [output_root / experiment_type]
    else:
        search_dirs = [d for d in output_root.iterdir() if d.is_dir()]

    for exp_dir in search_dirs:
        if not exp_dir.is_dir():
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if run_dir.is_dir() and (run_dir / "metadata").exists():
                capsules.append(RunCapsule(run_dir))

    return capsules


def _json_serializer(obj):
    """JSON serializer for numpy types and other non-standard objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)
