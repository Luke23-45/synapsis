from __future__ import annotations

import torch
from torch import nn


class EventEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.transition = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, structured_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, dim = structured_history.shape
        prev = torch.cat([structured_history[:, :1, :], structured_history[:, :-1, :]], dim=1)
        pair = torch.cat([structured_history, prev], dim=-1)
        hidden = self.transition(pair)
        scores = torch.relu(self.score_head(hidden).squeeze(-1))
        scores = scores.clone()
        scores[:, 0] = 0.0
        return hidden, scores
