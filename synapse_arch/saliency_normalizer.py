from __future__ import annotations

import torch
from torch import nn


class SaliencyNormalizer(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, event_scores: torch.Tensor) -> torch.Tensor:
        csum = torch.cumsum(event_scores, dim=1)
        csum_sq = torch.cumsum(event_scores * event_scores, dim=1)
        steps = torch.arange(1, event_scores.shape[1] + 1, device=event_scores.device, dtype=event_scores.dtype)
        mean = csum / steps.unsqueeze(0)
        var = torch.clamp(csum_sq / steps.unsqueeze(0) - mean * mean, min=0.0)
        std = torch.sqrt(var + self.eps)
        z = (event_scores - mean) / std
        temp = torch.exp(self.log_temperature).clamp(min=0.25, max=4.0)
        out = torch.sigmoid(z / temp) * event_scores
        out = out.clone()
        out[:, 0] = 0.0
        return out
