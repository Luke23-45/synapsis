"""
Structured configuration for the SYNAPSE Z2 memory operator experiments.

Z2 Reference: §5–9, §11, §13 of 02_rigorous_architecture.md
Formal Claims: Replaces Z1 tau/WeightsConfig with Z2 lam/W_Theta

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
    "RelaxedSelectorConfig",
    "NormalizationConfig",
    "GeometricLiftConfig",
    "TopologyConfig",
    "SaliencyConfig",
    "TrainingReadoutConfig",
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
# Z2 Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class RelaxedSelectorConfig:
    """Z2 relaxed selector parameters (§5 of 02_rigorous_architecture.md)."""
    K: int = 10
    r: int = 2
    lam: float = 1.0
    solver: str = "osqp"


@dataclass
class NormalizationConfig:
    """Z2 normalization parameters (§8 of 02_rigorous_architecture.md)."""
    mu: Optional[List[float]] = None
    sigma: Optional[List[float]] = None
    fit_from_data: bool = True


@dataclass
class GeometricLiftConfig:
    """Z2 learned lift parameters (§9 of 02_rigorous_architecture.md)."""
    k: int = 8
    init: str = "orthogonal"


@dataclass
class TopologyConfig:
    """Z2 topology parameters (§10 of 02_rigorous_architecture.md)."""
    Q: int = 1


@dataclass
class SaliencyConfig:
    """Z2 saliency normalizer (§4 of 02_rigorous_architecture.md)."""
    mode: str = "identity"
    temperature: float = 1.0


@dataclass
class TrainingReadoutConfig:
    """Z2 training/deployment readout split (§11 and §13)."""
    enabled: bool = False
    mode: str = "dense_soft_tokens"
    separate_training_head: bool = True


@dataclass
class MemoryOperatorConfig:
    """Z2 memory operator — composed from sub-configs."""
    selector: RelaxedSelectorConfig = field(default_factory=RelaxedSelectorConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    lift: GeometricLiftConfig = field(default_factory=GeometricLiftConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    saliency: SaliencyConfig = field(default_factory=SaliencyConfig)
    training_readout: TrainingReadoutConfig = field(default_factory=TrainingReadoutConfig)


# ---------------------------------------------------------------------------
# Shared sub-configs (Z1 + Z2 compatible)
# ---------------------------------------------------------------------------

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
            continue

        ft = field_types[key]
        if isinstance(ft, str):
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

    raw = _resolve_vars(raw, raw)

    # Build Z2 memory operator config from nested structure
    mo_data = raw.get("memory_operator", {})
    selector = _dict_to_dataclass(RelaxedSelectorConfig, mo_data.get("selector", {}))
    normalization = _dict_to_dataclass(NormalizationConfig, mo_data.get("normalization", {}))
    lift = _dict_to_dataclass(GeometricLiftConfig, mo_data.get("lift", {}))
    topology = _dict_to_dataclass(TopologyConfig, mo_data.get("topology", {}))
    saliency = _dict_to_dataclass(SaliencyConfig, mo_data.get("saliency", {}))
    training_readout = _dict_to_dataclass(TrainingReadoutConfig, mo_data.get("training_readout", {}))

    mo = MemoryOperatorConfig(
        selector=selector,
        normalization=normalization,
        lift=lift,
        topology=topology,
        saliency=saliency,
        training_readout=training_readout,
    )

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
        e.g. 'vz2_01_relaxed_selector'

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
    sel = mo.selector

    if sel.K < 1:
        raise ValueError(f"memory_operator.selector.K must be >= 1, got {sel.K}")
    if sel.r < 0:
        raise ValueError(f"memory_operator.selector.r must be >= 0, got {sel.r}")
    if sel.lam <= 0:
        raise ValueError(f"memory_operator.selector.lam must be > 0, got {sel.lam}")
    if sel.solver not in ("osqp", "scipy"):
        raise ValueError(
            f"memory_operator.selector.solver must be 'osqp' or 'scipy', got '{sel.solver}'"
        )

    norm = mo.normalization
    if norm.sigma is not None:
        for i, s in enumerate(norm.sigma):
            if s <= 0:
                raise ValueError(
                    f"memory_operator.normalization.sigma[{i}] must be > 0 (§8 invariant), got {s}"
                )

    lift_cfg = mo.lift
    if lift_cfg.k < 1:
        raise ValueError(f"memory_operator.lift.k must be >= 1, got {lift_cfg.k}")
    if lift_cfg.init not in ("orthogonal", "identity", "random"):
        raise ValueError(
            f"memory_operator.lift.init must be 'orthogonal', 'identity', or 'random', "
            f"got '{lift_cfg.init}'"
        )

    topo = mo.topology
    if topo.Q < 0:
        raise ValueError(f"memory_operator.topology.Q must be >= 0, got {topo.Q}")

    sal = mo.saliency
    if sal.mode not in ("identity", "z_score", "temperature"):
        raise ValueError(
            f"memory_operator.saliency.mode must be 'identity', 'z_score', or 'temperature', "
            f"got '{sal.mode}'"
        )
    if sal.mode == "temperature" and sal.temperature <= 0:
        raise ValueError(
            f"memory_operator.saliency.temperature must be > 0 for temperature mode, "
            f"got {sal.temperature}"
        )

    tr = mo.training_readout
    if tr.mode not in ("dense_soft_tokens", "proxy_topology", "custom"):
        raise ValueError(
            f"memory_operator.training_readout.mode must be 'dense_soft_tokens', "
            f"'proxy_topology', or 'custom', got '{tr.mode}'"
        )

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
