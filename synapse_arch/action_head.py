from __future__ import annotations

import torch
from torch import nn


class ActionHead(nn.Module):
    def __init__(self, d_model: int, action_chunk_size: int, action_dim: int) -> None:
        super().__init__()
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.head = nn.Linear(d_model, action_chunk_size * action_dim)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        flat = self.head(pooled)
        return flat.view(pooled.shape[0], self.action_chunk_size, self.action_dim)
