"""
Condition B: Transformer paired with cached SYNAPSE features.

The baseline experiment is intentionally not the full end-to-end training
stack from `synapse_arch`. It should consume the offline-cached anchor cloud
and topology summary produced by `src.synapse.synapse_cache`, project them
into token space, and feed those tokens to the shared transformer backbone.
"""

from __future__ import annotations

import torch

from src.core.config import Condition, ExperimentConfig, SynapseImplementation
from src.synapse import SynapseAdapter

from .planner import RoboticsPlannerBase
from .planner_synapse_e2e import PlannerSynapseEndToEnd


class PlannerSynapse(RoboticsPlannerBase):
    def __init__(self, config: ExperimentConfig) -> None:
        valid_conditions = {Condition.B_SYNAPSE, Condition.B_ANCHORS, Condition.B_TOPO}
        if config.condition not in valid_conditions:
            raise ValueError(f"PlannerSynapse requires a SYNAPSE condition, got {config.condition}")
        super().__init__(config)
        self.synapse_adapter = SynapseAdapter(config)

    @property
    def num_adapter_params(self) -> int:
        return self.synapse_adapter.num_trainable_params

    @property
    def parameter_overhead_pct(self) -> float:
        base = self.count_parameters_by_component()["total_base"]
        if base <= 0:
            return 0.0
        return 100.0 * self.num_adapter_params / base

    def count_parameters_by_component(self) -> dict:
        counts = super().count_parameters_by_component()
        counts["synapse_adapter"] = self.num_adapter_params
        counts["synapse_architecture"] = self.num_adapter_params
        counts["total_with_adapter"] = counts["total_base"] + counts["synapse_adapter"]
        return counts

    def forward(self, batch: dict) -> torch.Tensor:
        if "synapse_anchors" not in batch or "synapse_topo" not in batch:
            raise KeyError(
                "PlannerSynapse requires pre-computed 'synapse_anchors' and "
                "'synapse_topo' in the batch."
            )

        curr = self.proprio_encoder(batch["proprio"]).unsqueeze(1)
        synapse_tokens, n_synapse_tokens = self.synapse_adapter(
            batch["synapse_anchors"],
            batch["synapse_topo"],
        )
        tokens = torch.cat([curr, synapse_tokens], dim=1)
        return self._apply_transformer(tokens, n_real=1 + n_synapse_tokens)


def create_planner(config: ExperimentConfig):
    condition = config.condition
    if condition == Condition.A1_RECENT:
        from .planner_recent import PlannerRecent

        return PlannerRecent(config)
    if condition == Condition.A2_UNIFORM:
        from .planner_uniform import PlannerUniform

        return PlannerUniform(config)
    if condition in {Condition.B_SYNAPSE, Condition.B_ANCHORS, Condition.B_TOPO}:
        if config.synapse_implementation == SynapseImplementation.END_TO_END:
            return PlannerSynapseEndToEnd(config)
        return PlannerSynapse(config)
    raise ValueError(f"Unknown condition: {condition}")
