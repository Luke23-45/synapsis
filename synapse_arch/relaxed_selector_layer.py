from __future__ import annotations

import numpy as np
import torch
from torch import nn

from synapse_core.anchor_selector import solve_relaxed_selector


class _RelaxedSelectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, saliency_scores: torch.Tensor, K: int, r: int, lam: float, solver: str) -> torch.Tensor:
        ctx.lam = lam
        outputs = []
        saliency_np = saliency_scores.detach().cpu().numpy()
        for row in saliency_np:
            y_star = solve_relaxed_selector(
                np.asarray(row, dtype=np.float64), K, r, lam, solver=solver
            )
            outputs.append(torch.from_numpy(y_star).to(saliency_scores.device, dtype=saliency_scores.dtype))
        
        y_star_tensor = torch.stack(outputs, dim=0)
        ctx.save_for_backward(y_star_tensor)
        return y_star_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-Through Estimator (STE) proxy:
        # We pass the gradient directly through the selector to provide a stronger
        # training signal to the event encoder.
        return grad_output, None, None, None, None


class RelaxedSelectorLayer(nn.Module):
    def __init__(self, K: int, r: int, lam: float, solver: str = "osqp") -> None:
        super().__init__()
        self.K = K
        self.r = r
        self.lam = lam
        self.solver = solver

    def forward(self, saliency_scores: torch.Tensor) -> torch.Tensor:
        return _RelaxedSelectorFunction.apply(saliency_scores, self.K, self.r, self.lam, self.solver)
