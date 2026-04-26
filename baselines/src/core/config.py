"""
Experiment Configuration — Phase 4 Multi-Dataset Baselines
============================================================

Extended from the Phase 3 M1 config to support:
  - Multiple datasets (D1-D4) with per-dataset dimension overrides
  - Dataset-specific hyperparameter tuning (batch size, episode length)
  - Cross-domain experiment orchestration

Backward compatible: all Phase 3 defaults are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


class Condition(str, Enum):
    """Experimental conditions with controlled variable isolation."""

    A1_RECENT = "A1_recent"
    A2_UNIFORM = "A2_uniform"
    B_SYNAPSE = "B_synapse"
    B_ANCHORS = "B_anchors"
    B_TOPO = "B_topo"

    @property
    def label(self) -> str:
        return {
            Condition.A1_RECENT: "Recent Window",
            Condition.A2_UNIFORM: "Uniform Subsampling",
            Condition.B_SYNAPSE: "SYNAPSE Full",
            Condition.B_ANCHORS: "SYNAPSE Anchors Only",
            Condition.B_TOPO: "SYNAPSE Topology Only",
        }[self]

    @property
    def is_primary(self) -> bool:
        return self in (Condition.A1_RECENT, Condition.A2_UNIFORM, Condition.B_SYNAPSE)

    @property
    def is_ablation(self) -> bool:
        return self in (Condition.B_ANCHORS, Condition.B_TOPO)

    @property
    def uses_synapse(self) -> bool:
        return self in (
            Condition.B_SYNAPSE,
            Condition.B_ANCHORS,
            Condition.B_TOPO,
        )

    @property
    def use_anchors(self) -> bool:
        return self in (Condition.B_SYNAPSE, Condition.B_ANCHORS)

    @property
    def use_topo(self) -> bool:
        return self in (Condition.B_SYNAPSE, Condition.B_TOPO)

    @property
    def uses_full_horizon(self) -> bool:
        return self in (Condition.A2_UNIFORM, Condition.B_SYNAPSE, Condition.B_ANCHORS, Condition.B_TOPO)


@dataclass(frozen=True)
class SynapseParams:
    """Parameters for the SYNAPSE memory operator M().

    These are the mathematical parameters of the non-parametric operator.
    They are NOT learned — they are set before the experiment begins.
    """

    K: int = 10
    r: int = 3
    tau: float = 0.5
    lam: float = 0.5
    k: int = 32
    weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    Q: int = 1
    alpha: float = 0.0
    max_edge_length: Optional[float] = None


@dataclass(frozen=True)
class TransformerParams:
    """Shared transformer backbone architecture.

    Identical across all conditions — this is the controlled variable.
    """

    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ffn_ratio: int = 4
    dropout: float = 0.1
    activation: str = "gelu"


@dataclass(frozen=True)
class TrainingParams:
    """Training hyperparameters — identical across all conditions."""

    max_epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 500
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    use_amp: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    compile_model: bool = False
    fused_adamw: bool = True


@dataclass(frozen=True)
class DatasetSpec:
    """Per-dataset specification for multi-dataset experiments.

    Each dataset may have different dimensions, episode lengths, etc.
    These override the global DataParams when a specific dataset is active.
    """

    name: str = "pusht"
    source: str = "lerobot"                  # "lerobot" (HuggingFace)
    dataset_root: Optional[str] = None      # Root directory for canonical on-disk dataset layout
    local_path: Optional[str] = None        # Path to local Parquet file
    proprio_dim: int = 2
    action_dim: int = 2
    max_episode_length: int = 300
    num_phases: int = 3
    fps: int = 10
    batch_size_override: Optional[int] = None  # Override global batch_size
    max_episodes: Optional[int] = None         # Limit episodes (for testing)


@dataclass(frozen=True)
class DataParams:
    """Global data dimensions and pipeline configuration.

    These are defaults that can be overridden by DatasetSpec.
    """

    proprio_dim: int = 14
    ee_pose_dim: int = 7
    ee_vel_dim: int = 6
    object_pos_dim: int = 3
    grasp_dim: int = 1
    action_dim: int = 14
    action_chunk_size: int = 8
    history_window: int = 50
    max_episode_length: int = 400
    num_phases: int = 4
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass(frozen=True)
class StatsParams:
    """Statistical rigor parameters for the comparison protocol."""

    num_seeds_primary: int = 5
    num_seeds_ablation: int = 3
    significance_level: float = 0.05
    confidence_level: float = 0.95
    num_bootstrap_samples: int = 10000
    rollout_steps: int = 10


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete experiment configuration.

    Extended for Phase 4 with multi-dataset support via `datasets` field.
    """

    condition: Condition = Condition.B_SYNAPSE
    seed: int = 42
    synapse: SynapseParams = field(default_factory=SynapseParams)
    transformer: TransformerParams = field(default_factory=TransformerParams)
    training: TrainingParams = field(default_factory=TrainingParams)
    data: DataParams = field(default_factory=DataParams)
    stats: StatsParams = field(default_factory=StatsParams)
    output_dir: str = "output"
    experiment_name: str = "m1_utility_proof"

    # Phase 4: Multi-dataset support (public HuggingFace benchmarks)
    datasets: Tuple[DatasetSpec, ...] = field(default_factory=lambda: (
        DatasetSpec(name="pusht", source="lerobot", proprio_dim=2, action_dim=2, max_episode_length=300, num_phases=3),
    ))

    @property
    def seq_len(self) -> int:
        """Fixed sequence length for token-count equalization (§3.6)."""
        return self.data.history_window + 1

    @property
    def memory_K(self) -> int:
        """Anchor budget — convenience accessor."""
        return self.synapse.K

    @property
    def memory_Q(self) -> int:
        """Max homology degree — convenience accessor."""
        return self.synapse.Q

    @property
    def topo_feature_dim(self) -> int:
        """Dimensionality of the topological summary vector.

        4 statistics × (Q+1) diagram degrees = 4(Q+1).
        """
        return 4 * (self.synapse.Q + 1)

    @property
    def anchor_feature_dim(self) -> int:
        """Dimensionality of each lifted anchor point.

        proprio_dim + 3 (t, delta, xi from the geometric lift ρ).
        """
        return self.data.proprio_dim + 3

    @property
    def structured_state_dim(self) -> int:
        return (
            self.data.proprio_dim
            + self.data.ee_pose_dim
            + self.data.ee_vel_dim
            + self.data.object_pos_dim
            + self.data.grasp_dim
        )

    def for_dataset(self, dataset_spec: DatasetSpec) -> "ExperimentConfig":
        """Create a dataset-specific config by overriding dimensions.

        This is critical for multi-dataset experiments where each dataset
        has different proprio_dim, action_dim, etc.
        """
        overridden_data = DataParams(
            proprio_dim=dataset_spec.proprio_dim,
            action_dim=dataset_spec.action_dim,
            action_chunk_size=self.data.action_chunk_size,
            history_window=self.data.history_window,
            max_episode_length=dataset_spec.max_episode_length,
            num_phases=dataset_spec.num_phases,
            train_ratio=self.data.train_ratio,
            val_ratio=self.data.val_ratio,
            test_ratio=self.data.test_ratio,
            ee_pose_dim=self.data.ee_pose_dim,
            ee_vel_dim=self.data.ee_vel_dim,
            object_pos_dim=self.data.object_pos_dim,
            grasp_dim=self.data.grasp_dim,

        )

        overridden_training = self.training
        if dataset_spec.batch_size_override is not None:
            overridden_training = TrainingParams(
                max_epochs=self.training.max_epochs,
                batch_size=dataset_spec.batch_size_override,
                learning_rate=self.training.learning_rate,
                weight_decay=self.training.weight_decay,
                beta1=self.training.beta1,
                beta2=self.training.beta2,
                warmup_steps=self.training.warmup_steps,
                early_stopping_patience=self.training.early_stopping_patience,
                gradient_clip_norm=self.training.gradient_clip_norm,
                use_amp=self.training.use_amp,
                num_workers=self.training.num_workers,
                pin_memory=self.training.pin_memory,
                persistent_workers=self.training.persistent_workers,
                prefetch_factor=self.training.prefetch_factor,
                compile_model=self.training.compile_model,
                fused_adamw=self.training.fused_adamw,
            )

        return ExperimentConfig(
            condition=self.condition,
            seed=self.seed,
            synapse=self.synapse,
            transformer=self.transformer,
            training=overridden_training,
            data=overridden_data,
            stats=self.stats,
            output_dir=self.output_dir,
            experiment_name=self.experiment_name,
            datasets=self.datasets,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["condition"] = self.condition.value
        d["synapse"]["weights"] = list(self.synapse.weights)
        # Convert DatasetSpec tuples
        d["datasets"] = [asdict(ds) for ds in self.datasets]
        return d

    def save_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from a YAML file.

    The YAML file may specify any subset of fields; unspecified fields
    retain their PLAN.md defaults.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return _config_from_dict(raw)


def _config_from_dict(raw: dict) -> ExperimentConfig:
    """Construct ExperimentConfig from a nested dictionary."""

    def _pop_nested(d: dict, key: str, default_factory):
        sub = d.pop(key, {})
        if sub is None:
            sub = {}
        return default_factory(**sub)

    condition_str = raw.pop("condition", "B_synapse")
    condition = Condition(condition_str)
    seed = raw.pop("seed", 42)
    output_dir = raw.pop("output_dir", "output")
    experiment_name = raw.pop("experiment_name", "m1_utility_proof")

    synapse_raw = raw.pop("synapse", {}) or {}
    if "weights" in synapse_raw:
        synapse_raw["weights"] = tuple(synapse_raw["weights"])
    synapse = SynapseParams(**synapse_raw)

    transformer = _pop_nested(raw, "transformer", TransformerParams)
    training = _pop_nested(raw, "training", TrainingParams)
    data = _pop_nested(raw, "data", DataParams)
    stats = _pop_nested(raw, "stats", StatsParams)

    # Parse datasets list
    datasets_raw = raw.pop("datasets", None)
    if datasets_raw is not None:
        datasets = tuple(DatasetSpec(**ds) for ds in datasets_raw)
    else:
        datasets = (DatasetSpec(name="pusht", source="lerobot"),)

    if raw:
        import warnings
        warnings.warn(f"Unrecognized config fields ignored: {list(raw.keys())}")

    return ExperimentConfig(
        condition=condition,
        seed=seed,
        synapse=synapse,
        transformer=transformer,
        training=training,
        data=data,
        stats=stats,
        output_dir=output_dir,
        experiment_name=experiment_name,
        datasets=datasets,
    )
