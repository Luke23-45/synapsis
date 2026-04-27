from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
from torch import nn

from synapse_core.topological_summary import compute_persistence_diagrams


def summarize_diagrams(diagrams: List[Any]) -> np.ndarray:
    summary: list[float] = []
    for dgm in diagrams:
        points = np.asarray(dgm, dtype=np.float64)
        if points.size == 0:
            summary.extend([0.0, 0.0, 0.0, 0.0])
            continue
        if points.ndim == 1:
            points = points.reshape(-1, 2)
        finite = points[np.isfinite(points[:, 1])]
        if finite.size == 0:
            summary.extend([0.0, 0.0, 0.0, 0.0])
            continue
        persistence = finite[:, 1] - finite[:, 0]
        summary.extend([
            float(len(finite)),
            float(np.mean(persistence)),
            float(np.max(persistence)),
            float(np.sum(persistence)),
        ])
    return np.asarray(summary, dtype=np.float32)


class TopologyBranch(nn.Module):
    def __init__(self, lift_dim: int, summary_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.summary_dim = summary_dim
        self.proj = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lift_dim = lift_dim

    def _surrogate_summary(
        self,
        lifted_tokens: torch.Tensor,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        def _safe_norm(x: torch.Tensor, dim: int) -> torch.Tensor:
            return torch.sqrt(x.square().sum(dim=dim) + 1e-6)

        weights = activations.unsqueeze(-1)
        weighted = lifted_tokens * weights
        mass = activations.sum(dim=1, keepdim=True)
        denom = mass.clamp_min(1.0)
        centroid = weighted.sum(dim=1) / denom
        diffs = lifted_tokens - centroid.unsqueeze(1)

        pairwise_diffs = lifted_tokens.unsqueeze(2) - lifted_tokens.unsqueeze(1)
        pairwise = _safe_norm(pairwise_diffs, dim=-1)
        pair_weights = torch.matmul(activations.unsqueeze(-1), activations.unsqueeze(1))
        tri_mask = torch.triu(torch.ones_like(pairwise), diagonal=1)
        tri_weights = pair_weights * tri_mask
        weighted_pairwise = pairwise * tri_weights
        pair_mass = tri_weights.sum(dim=(1, 2)).clamp_min(1.0)

        mean_pair = weighted_pairwise.sum(dim=(1, 2)) / pair_mass
        max_pair = weighted_pairwise.amax(dim=(1, 2))
        pair_centered = (pairwise - mean_pair[:, None, None]) * tri_weights
        pair_var = (pair_centered.square().sum(dim=(1, 2)) / pair_mass).clamp_min(0.0)
        pair_std = torch.sqrt(pair_var + 1e-6)

        disp = _safe_norm(diffs, dim=-1)
        weighted_disp = disp * activations
        mean_disp = weighted_disp.sum(dim=1) / denom.squeeze(1)
        max_disp = weighted_disp.amax(dim=1)
        disp_centered = (disp - mean_disp[:, None]) * activations
        disp_var = (disp_centered.square().sum(dim=1) / denom.squeeze(1)).clamp_min(0.0)
        disp_std = torch.sqrt(disp_var + 1e-6)

        support_ratio = (activations > 1e-3).float().mean(dim=1)
        activation_mean = activations.mean(dim=1)
        activation_std = activations.std(dim=1, unbiased=False)
        centroid_norm = _safe_norm(centroid, dim=1)
        token_norm_mean = (_safe_norm(lifted_tokens, dim=-1) * activations).sum(dim=1) / denom.squeeze(1)
        second_moment = torch.sqrt(weighted.square().sum(dim=(1, 2)) / denom.squeeze(1) + 1e-6)

        features = torch.stack(
            [
                mean_pair,
                max_pair,
                pair_std,
                mean_disp,
                max_disp,
                disp_std,
                support_ratio,
                activation_mean,
                activation_std,
                centroid_norm,
                token_norm_mean,
                second_moment,
            ],
            dim=-1,
        )
        if self.summary_dim <= features.shape[1]:
            return features[:, : self.summary_dim]
        pad = torch.zeros(
            features.shape[0],
            self.summary_dim - features.shape[1],
            device=features.device,
            dtype=features.dtype,
        )
        return torch.cat([features, pad], dim=-1)

    def surrogate(self, lifted_tokens: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
        summary = self._surrogate_summary(lifted_tokens, activations)
        return self.proj(summary)

    def exact(self, point_cloud: np.ndarray, Q: int, max_edge_length: float | None = None) -> tuple[List[Any], np.ndarray]:
        diagrams = compute_persistence_diagrams(point_cloud, Q, max_edge_length=max_edge_length)
        return diagrams, summarize_diagrams(diagrams)
