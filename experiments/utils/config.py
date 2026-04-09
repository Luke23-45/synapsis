"""
Structured configuration for the SYNAPSE memory operator experiments.

Adapted from GibbsQ's config architecture: typed dataclass schemas
loaded from YAML, with comprehensive validation so invalid configurations
are caught before experiment execution.

Usage
-----
::

    from experiments.utils.config import load_config, validate

    cfg = load_config("experiments/configs/default.yaml")
    validate(cfg)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


__all__ = [
    "MemoryOperatorConfig",
    "EncoderConfig",
    "TrajectoryConfig",
    "ExecutionConfig",
    "VerificationConfig",
    "PlottingConfig",
    "ExperimentConfig",
    "load_config",
    "validate",
    "get_experiment_overrides",
]


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class WeightsConfig:
    """Geometric lift weights (w_t, w_x, w_δ, w_e)."""
    w_t: float = 1.0
    w_x: float = 1.0
    w_delta: float = 1.0
    w_e: float = 1.0

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.w_t, self.w_x, self.w_delta, self.w_e)


@dataclass
class MemoryOperatorConfig:
    """
    Core memory operator parameters.

    Fields mirror §5–§8 of 01_main_definition.md:
        K   — maximum anchor budget
        r   — minimum refractory separation
        tau — event score threshold
        Q   — maximum homology degree
    """
    K: int = 10
    r: int = 2
    tau: float = 0.5
    Q: int = 1
    weights: WeightsConfig = field(default_factory=WeightsConfig)


@dataclass
class EncoderConfig:
    """Event encoder mode selection."""
    mode: str = "sharp"    # 'sharp' or 'hysteretic'
    alpha: float = 0.0     # Hysteresis gate


@dataclass
class TrajectoryConfig:
    """Default trajectory generation parameters."""
    d: int = 5
    T: int = 100
    seed: int = 42


@dataclass
class ExecutionConfig:
    """Experiment execution parameters."""
    num_replications: int = 50
    seed: int = 42
    verbose: bool = False
    strict: bool = False


@dataclass
class VerificationConfig:
    """Pass/fail thresholds for experiment assertions."""
    score_match_atol: float = 1e-14
    reconstruction_atol: float = 1e-14
    hausdorff_rtol: float = 1e-10
    bottleneck_rtol: float = 1e-10
    history_diff_threshold: float = 1e-10


@dataclass
class PlottingConfig:
    """Plotting output configuration."""
    theme: str = "publication"
    formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    dpi: int = 600


@dataclass
class ExperimentConfig:
    """
    Root configuration node.

    Composed from sub-configs plus output paths and
    per-experiment override dictionaries.
    """
    memory_operator: MemoryOperatorConfig = field(default_factory=MemoryOperatorConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    experiments: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = "experiments/outputs"
    log_dir: str = "experiments/outputs/logs"


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _resolve_vars(data: Any, root: dict) -> Any:
    """Resolve ${var} references in string values."""
    if isinstance(data, str) and "${" in data:
        for key, val in root.items():
            if isinstance(val, str):
                data = data.replace(f"${{{key}}}", val)
        return data
    if isinstance(data, dict):
        return {k: _resolve_vars(v, root) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_vars(v, root) for v in data]
    return data


def _dict_to_dataclass(cls, data: dict):
    """Recursively instantiate a dataclass from a nested dict."""
    if data is None:
        return cls()

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key not in field_types:
            # Unknown key — store in experiments dict if root level
            continue

        ft = field_types[key]
        # Check if the field type is itself a dataclass
        if isinstance(ft, str):
            # Handle forward references
            ft = globals().get(ft, str)

        if isinstance(ft, type) and hasattr(ft, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(ft, value)
        else:
            kwargs[key] = value

    return cls(**kwargs)


def load_config(
    config_path: str | Path,
    overrides: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    """
    Load experiment configuration from a YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file.
    overrides : dict, optional
        Key-value overrides applied after loading.

    Returns
    -------
    ExperimentConfig
        Validated configuration object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    # Resolve variable references
    raw = _resolve_vars(raw, raw)

    # Build sub-configs
    mo_data = raw.get("memory_operator", {})
    weights_data = mo_data.pop("weights", {}) if isinstance(mo_data, dict) else {}
    weights = _dict_to_dataclass(WeightsConfig, weights_data)

    mo = _dict_to_dataclass(MemoryOperatorConfig, mo_data)
    mo.weights = weights

    encoder = _dict_to_dataclass(EncoderConfig, raw.get("encoder", {}))
    trajectory = _dict_to_dataclass(TrajectoryConfig, raw.get("trajectory", {}))
    execution = _dict_to_dataclass(ExecutionConfig, raw.get("execution", {}))
    verification = _dict_to_dataclass(VerificationConfig, raw.get("verification", {}))
    plotting = _dict_to_dataclass(PlottingConfig, raw.get("plotting", {}))

    cfg = ExperimentConfig(
        memory_operator=mo,
        encoder=encoder,
        trajectory=trajectory,
        execution=execution,
        verification=verification,
        plotting=plotting,
        experiments=raw.get("experiments", {}),
        output_dir=raw.get("output_dir", "experiments/outputs"),
        log_dir=raw.get("log_dir", "experiments/outputs/logs"),
    )

    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            obj = cfg
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

    return cfg


def get_experiment_overrides(
    cfg: ExperimentConfig,
    experiment_id: str,
) -> Dict[str, Any]:
    """
    Get per-experiment overrides from the config.

    Parameters
    ----------
    cfg : ExperimentConfig
    experiment_id : str
        e.g. 'exp01_causality'

    Returns
    -------
    dict
        Override values for this experiment, or empty dict.
    """
    return cfg.experiments.get(experiment_id, {})


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(cfg: ExperimentConfig) -> None:
    """
    Validate every constraint on *cfg* and raise ``ValueError`` with a
    precise diagnostic on the first violation.

    Called at the beginning of every experiment script.
    """
    mo = cfg.memory_operator

    if mo.K < 0:
        raise ValueError(f"memory_operator.K must be >= 0, got {mo.K}")
    if mo.r < 0:
        raise ValueError(f"memory_operator.r must be >= 0, got {mo.r}")
    if mo.tau <= 0:
        raise ValueError(f"memory_operator.tau must be > 0, got {mo.tau}")
    if mo.Q < 0:
        raise ValueError(f"memory_operator.Q must be >= 0, got {mo.Q}")

    w = mo.weights
    for name, val in [("w_t", w.w_t), ("w_x", w.w_x), ("w_delta", w.w_delta), ("w_e", w.w_e)]:
        if val <= 0:
            raise ValueError(f"memory_operator.weights.{name} must be > 0, got {val}")

    enc = cfg.encoder
    if enc.mode not in ("sharp", "hysteretic"):
        raise ValueError(
            f"encoder.mode must be 'sharp' or 'hysteretic', got '{enc.mode}'"
        )
    if enc.mode == "hysteretic" and not (0.0 <= enc.alpha < 1.0):
        raise ValueError(
            f"encoder.alpha must be in [0, 1) for hysteretic mode, got {enc.alpha}"
        )

    traj = cfg.trajectory
    if traj.d < 1:
        raise ValueError(f"trajectory.d must be >= 1, got {traj.d}")
    if traj.T < 1:
        raise ValueError(f"trajectory.T must be >= 1, got {traj.T}")

    exe = cfg.execution
    if exe.num_replications < 1:
        raise ValueError(
            f"execution.num_replications must be >= 1, got {exe.num_replications}"
        )

    ver = cfg.verification
    for name in ["score_match_atol", "reconstruction_atol", "hausdorff_rtol", "bottleneck_rtol"]:
        val = getattr(ver, name)
        if val <= 0:
            raise ValueError(f"verification.{name} must be > 0, got {val}")

    plot = cfg.plotting
    valid_themes = {"publication", "dark"}
    if plot.theme not in valid_themes:
        raise ValueError(
            f"plotting.theme must be one of {sorted(valid_themes)}, got '{plot.theme}'"
        )
    if plot.dpi < 72:
        raise ValueError(f"plotting.dpi must be >= 72, got {plot.dpi}")

    if not cfg.output_dir.strip():
        raise ValueError("output_dir must be a non-empty string")
