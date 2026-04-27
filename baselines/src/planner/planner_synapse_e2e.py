from __future__ import annotations

import torch
from torch import nn

from src.core.config import Condition, ExperimentConfig
from synapse_arch.model import SynapseArchitectureConfig, SynapseEndToEndModel


class PlannerSynapseEndToEnd(nn.Module):
    """Adapter that exposes the end-to-end SYNAPSE model via the baseline API."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        valid_conditions = {Condition.B_SYNAPSE, Condition.B_ANCHORS, Condition.B_TOPO}
        if config.condition not in valid_conditions:
            raise ValueError(
                f"PlannerSynapseEndToEnd requires a SYNAPSE condition, got {config.condition}"
            )
        self.config = config
        arch_config = SynapseArchitectureConfig(
            input_dim=config.structured_state_dim,
            action_dim=config.data.action_dim,
            action_chunk_size=config.data.action_chunk_size,
            hidden_dim=config.transformer.event_encoder_hidden_dim,
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
            keep_all_anchors=config.synapse.keep_all_anchors,
        )
        self.architecture = SynapseEndToEndModel(arch_config)

    @property
    def num_trainable_params(self) -> int:
        return self.architecture.num_trainable_params

    def count_parameters_by_component(self) -> dict:
        return {
            "synapse_architecture": self.num_trainable_params,
            "total_base": self.num_trainable_params,
            "total_with_adapter": self.num_trainable_params,
        }

    def forward_train(self, batch: dict):
        return self.architecture.forward_train(
            batch,
            use_anchors=self.config.condition.use_anchors,
            use_topology=self.config.condition.use_topo,
        )

    def forward_deploy(self, batch: dict):
        return self.architecture.forward_deploy(
            batch,
            use_anchors=self.config.condition.use_anchors,
            use_topology=self.config.condition.use_topo,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.forward_train(batch).pred_actions
