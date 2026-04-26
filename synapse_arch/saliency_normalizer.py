from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SaliencyNormalizer(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, event_scores: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if padding_mask is None:
            padding_mask = torch.ones_like(event_scores, dtype=torch.bool)
            
        valid_scores = event_scores * padding_mask.type_as(event_scores)
        
        csum = torch.cumsum(valid_scores, dim=1)
        csum_sq = torch.cumsum(valid_scores * valid_scores, dim=1)
        
        steps = torch.cumsum(padding_mask.type_as(event_scores), dim=1).clamp_min(1.0)
        
        mean = csum / steps
        var = torch.clamp(csum_sq / steps - mean * mean, min=0.0)
        std = torch.sqrt(var + self.eps)
        z = (event_scores - mean) / std
        temp = torch.exp(self.log_temperature).clamp(min=0.25, max=4.0)
        
        # Saliency must be non-negative for the relaxed selector. Raw event
        # logits can be negative, so convert magnitude with softplus and gate
        # it by the causal z-score confidence.
        out = torch.sigmoid(z / temp) * F.softplus(event_scores)
        out = out.clone()
        out = out.masked_fill(~padding_mask, 0.0)
        out[:, 0] = -1e9  # Mask applied safely after normalization
        return out
