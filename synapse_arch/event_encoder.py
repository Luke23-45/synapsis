from __future__ import annotations

import torch
from torch import nn


class EventEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.transition = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, structured_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, dim = structured_history.shape
        prev = torch.cat([structured_history[:, :1, :], structured_history[:, :-1, :]], dim=1)
        diff = structured_history - prev
        pair = torch.cat([structured_history, prev, diff], dim=-1)
        hidden = self.transition(pair)
        # Output raw logits for BCEWithLogitsLoss
        scores = self.score_head(hidden).squeeze(-1)
        return hidden, scores
