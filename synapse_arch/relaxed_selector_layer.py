from __future__ import annotations

import numpy as np
import torch
from torch import nn

from synapse_core.anchor_selector import solve_relaxed_selector


class RelaxedSelectorLayer(nn.Module):
    def __init__(self, K: int, r: int, lam: float, solver: str = "scipy") -> None:
        super().__init__()
        self.K = K
        self.r = r
        self.lam = lam
        self.solver = solver

    def forward(self, saliency_scores: torch.Tensor) -> torch.Tensor:
        outputs = []
        saliency_np = saliency_scores.detach().cpu().numpy()
        for row in saliency_np:
            y_star = solve_relaxed_selector(np.asarray(row, dtype=np.float64), self.K, self.r, self.lam, solver=self.solver)
            outputs.append(torch.from_numpy(y_star).to(saliency_scores.device, dtype=saliency_scores.dtype))
        return torch.stack(outputs, dim=0)
