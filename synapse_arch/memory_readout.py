from __future__ import annotations

import torch
from torch import nn


class MemoryReadout(nn.Module):
    def __init__(self, lift_dim: int, d_model: int, topology_dim: int) -> None:
        super().__init__()
        self.anchor_proj = nn.Linear(lift_dim, d_model)
        self.topology_proj = nn.Linear(topology_dim, d_model)

    def forward_train(self, dense_lifted_tokens: torch.Tensor, topology_features: torch.Tensor, activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_tokens = self.anchor_proj(dense_lifted_tokens) * activations.unsqueeze(-1)
        topo_token = self.topology_proj(topology_features).unsqueeze(1)
        return anchor_tokens, topo_token

    def forward_deploy(self, anchor_cloud: torch.Tensor, topology_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_tokens = self.anchor_proj(anchor_cloud)
        topo_token = self.topology_proj(topology_features).unsqueeze(1)
        return anchor_tokens, topo_token
