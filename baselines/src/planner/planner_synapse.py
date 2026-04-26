"""
Condition B: End-to-end trainable SYNAPSE architecture.

This planner delegates the full train/deploy memory pipeline to the
top-level `synapse_arch` package and keeps `applied_robotics` responsible
for data, orchestration, evaluation, and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from synapse_arch import SynapseEndToEndModel
from synapse_arch.model import SynapseArchitectureConfig

from src.core.config import Condition, ExperimentConfig


@dataclass
class _CompatibilityAdapter:
    use_anchors: bool
    use_topo: bool
    num_output_tokens: int


class PlannerSynapse(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        valid_conditions = {Condition.B_SYNAPSE, Condition.B_ANCHORS, Condition.B_TOPO}
        if config.condition not in valid_conditions:
            raise ValueError(f"PlannerSynapse requires a SYNAPSE condition, got {config.condition}")
        self.config = config
        self.condition = config.condition
        self.seq_len = config.seq_len
        self.architecture = SynapseEndToEndModel(
            SynapseArchitectureConfig(
                input_dim=config.structured_state_dim,
                action_dim=config.data.action_dim,
                action_chunk_size=config.data.action_chunk_size,
                hidden_dim=config.transformer.d_model,
                d_model=config.transformer.d_model,
                num_heads=config.transformer.num_heads,
                num_layers=config.transformer.num_layers,
                ffn_ratio=config.transformer.ffn_ratio,
                dropout=config.transformer.dropout,
                K=config.synapse.K,
                r=config.synapse.r,
                lam=config.synapse.lam,
                Q=config.synapse.Q,
                k=config.synapse.k,
                max_history_tokens=config.data.max_episode_length,
            )
        )
        self.synapse_adapter = _CompatibilityAdapter(
            use_anchors=config.condition.use_anchors,
            use_topo=config.condition.use_topo,
            num_output_tokens=(config.synapse.K if config.condition.use_anchors else 0) + (1 if config.condition.use_topo else 0),
        )

    @property
    def num_trainable_params(self) -> int:
        return self.architecture.num_trainable_params

    @property
    def num_adapter_params(self) -> int:
        return self.architecture.num_trainable_params

    @property
    def parameter_overhead_pct(self) -> float:
        return 0.0

    def count_parameters_by_component(self) -> dict:
        return {
            "synapse_architecture": self.num_trainable_params,
            "total_base": self.num_trainable_params,
        }

    def forward_train(self, batch: dict):
        return self.architecture.forward_train(
            batch,
            use_anchors=self.condition.use_anchors,
            use_topology=self.condition.use_topo,
        )

    def forward_deploy(self, batch: dict):
        return self.architecture.forward_deploy(
            batch,
            use_anchors=self.condition.use_anchors,
            use_topology=self.condition.use_topo,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.forward_train(batch).pred_actions


def create_planner(config: ExperimentConfig):
    condition = config.condition
    if condition == Condition.A1_RECENT:
        from .planner_recent import PlannerRecent

        return PlannerRecent(config)
    if condition == Condition.A2_UNIFORM:
        from .planner_uniform import PlannerUniform

        return PlannerUniform(config)
    if condition in {Condition.B_SYNAPSE, Condition.B_ANCHORS, Condition.B_TOPO}:
        return PlannerSynapse(config)
    raise ValueError(f"Unknown condition: {condition}")
