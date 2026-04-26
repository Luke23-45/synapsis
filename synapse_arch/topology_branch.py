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

    def surrogate(self, lifted_tokens: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
        weights = activations.unsqueeze(-1)
        weighted = lifted_tokens * weights
        # Clamping denom prevents division blowups for highly sparse selectors,
        # while still driving the summary toward zero when no anchors are active.
        mass = activations.sum(dim=1, keepdim=True)
        denom = mass.clamp_min(1.0)
        centroid = weighted.sum(dim=1) / denom
        diffs = lifted_tokens - centroid.unsqueeze(1)
        pairwise = torch.cdist(lifted_tokens, lifted_tokens)
        pair_weights = torch.matmul(activations.unsqueeze(-1), activations.unsqueeze(1))
        weighted_pairwise = pairwise * pair_weights
        tri_mask = torch.triu(torch.ones_like(weighted_pairwise), diagonal=1)
        tri = weighted_pairwise * tri_mask
        pair_mass = (pair_weights * tri_mask).sum(dim=(1, 2)).clamp_min(1.0)
        mean_pair = tri.sum(dim=(1, 2)) / pair_mass
        max_pair = tri.amax(dim=(1, 2))
        disp = diffs.norm(dim=-1)
        weighted_disp = disp * activations
        mean_disp = weighted_disp.sum(dim=1) / denom.squeeze(1)
        max_disp = weighted_disp.amax(dim=1)
        summary = torch.stack([mean_pair, max_pair, mean_disp, max_disp], dim=-1)
        if self.summary_dim > 4:
            repeats = self.summary_dim // 4
            summary = summary.repeat(1, repeats)
        return self.proj(summary)

    def exact(self, point_cloud: np.ndarray, Q: int, max_edge_length: float | None = None) -> tuple[List[Any], np.ndarray]:
        diagrams = compute_persistence_diagrams(point_cloud, Q, max_edge_length=max_edge_length)
        return diagrams, summarize_diagrams(diagrams)
