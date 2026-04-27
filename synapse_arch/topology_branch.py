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
    NUM_SPECTRAL_SCALES = 4
    NUM_SPECTRAL_EIGVALS = 4

    def __init__(self, lift_dim: int, summary_dim: int, hidden_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.summary_dim = summary_dim
        self.num_heads = num_heads
        self.proj = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.surrogate_proj = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Differentiable spectral geometry features (replaces detached Gudhi)
        spectral_feature_dim = self.NUM_SPECTRAL_SCALES * self.NUM_SPECTRAL_EIGVALS + 4
        self.log_scales = nn.Parameter(torch.linspace(-1.5, 1.5, self.NUM_SPECTRAL_SCALES))
        self.spectral_proj = nn.Sequential(
            nn.Linear(spectral_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Topological Attention Bias (TAB): per-head learnable scales
        # Each head attends at a different topological resolution
        # Head 0: small σ → local neighborhood attention
        # Head H: large σ → global manifold attention
        self.log_sigma_tab = nn.Parameter(torch.linspace(-1.0, 2.0, num_heads))
        # Learnable strength of topological bias per head
        self.tab_alpha = nn.Parameter(torch.ones(num_heads) * 0.5)
        self.lift_dim = lift_dim

    def compute_attention_bias(
        self,
        point_cloud: torch.Tensor,
        activations: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Compute topological attention bias from the lifted point cloud.

        Returns a (B, num_heads, seq_len, seq_len) bias matrix that is ADDED
        to the Transformer's attention logits before softmax. This forces the
        model to attend along the topological manifold rather than freely
        pattern-matching tokens.

        Parameters
        ----------
        point_cloud : (B, N, k)  time-invariant lifted anchor positions
        activations : (B, N)     anchor selection weights
        seq_len     : int        total token sequence length (1 + N for [current, anchors])

        Returns
        -------
        attn_bias : (B * num_heads, seq_len, seq_len)
        """
        B, N, k = point_cloud.shape

        # Compute pairwise distances (fully differentiable)
        cloud_f32 = point_cloud.float()
        D = torch.cdist(cloud_f32, cloud_f32)  # (B, N, N)

        # Per-head Gaussian affinity in log-space
        sigma = torch.exp(self.log_sigma_tab.float()).clamp(min=0.01)  # (num_heads,)
        alpha = self.tab_alpha.float()  # (num_heads,)

        # D: (B, N, N) → (B, 1, N, N)
        D_sq = D.square().unsqueeze(1)
        # sigma: (H,) → (1, H, 1, 1)
        sigma_sq = sigma.square().view(1, self.num_heads, 1, 1)
        alpha_h = alpha.view(1, self.num_heads, 1, 1)

        # Gaussian affinity: high value = "attend here", near-zero = "ignore"
        # This is added to QK^T/sqrt(d), so positive values encourage attention
        anchor_bias = alpha_h * torch.exp(-D_sq / (2.0 * sigma_sq + 1e-8))  # (B, H, N, N)

        # Mask out inactive anchors
        mask = (activations.float() > 1e-3)  # (B, N)
        mask_2d = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, N, N)
        mask_2d = mask_2d.unsqueeze(1).expand_as(anchor_bias)  # (B, H, N, N)
        # Inactive positions get large negative bias (effectively -inf after softmax)
        anchor_bias = anchor_bias.masked_fill(~mask_2d, -1e4)

        # Build full bias matrix including current token (position 0)
        # current token has no topological bias (attends freely to all anchors)
        full_bias = torch.zeros(B, self.num_heads, seq_len, seq_len,
                                device=point_cloud.device, dtype=torch.float32)

        # anchor-to-anchor region: rows 1..N+1, cols 1..N+1
        n_anchors = min(N, seq_len - 1)
        full_bias[:, :, 1:1+n_anchors, 1:1+n_anchors] = anchor_bias[:, :, :n_anchors, :n_anchors]

        # Reshape for PyTorch's attn_mask: (B*H, seq_len, seq_len)
        return full_bias.reshape(B * self.num_heads, seq_len, seq_len).to(point_cloud.dtype)

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
        return features

    def surrogate(self, lifted_tokens: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
        summary = self._surrogate_summary(lifted_tokens, activations)
        return self.surrogate_proj(summary)

    def exact(self, point_cloud: np.ndarray, Q: int, max_edge_length: float | None = None) -> tuple[List[Any], np.ndarray]:
        diagrams = compute_persistence_diagrams(point_cloud, Q, max_edge_length=max_edge_length)
        return diagrams, summarize_diagrams(diagrams)

    def spectral_features(self, point_cloud: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
        """Compute differentiable spectral topology features from the lifted point cloud.

        Uses eigenvalues of the graph Laplacian at multiple learnable scales to
        approximate persistent homology — fully differentiable so the action loss
        can directly shape the point cloud geometry.

        Parameters
        ----------
        point_cloud : (B, N, k)  lifted anchor positions
        activations : (B, N)     anchor activation weights

        Returns
        -------
        (B, d_model) projected spectral features
        """
        B, N, k = point_cloud.shape

        # Force float32 for eigvalsh numerical stability
        cloud_f32 = point_cloud.float()
        act_f32 = activations.float()

        mask = (act_f32 > 1e-3).float()
        mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)

        D = torch.cdist(cloud_f32, cloud_f32)
        D = D * mask_2d

        scales = torch.exp(self.log_scales.float())
        num_eigvals = self.NUM_SPECTRAL_EIGVALS

        features = []
        for s_idx in range(self.NUM_SPECTRAL_SCALES):
            sigma = scales[s_idx]
            A = torch.exp(-D.square() / (2 * sigma.square() + 1e-8))
            A = A * mask_2d

            degree = A.sum(dim=-1)
            L = torch.diag_embed(degree) - A
            L = L + torch.eye(N, device=L.device).unsqueeze(0) * 1e-6

            eigvals = torch.linalg.eigvalsh(L)
            features.append(eigvals[:, :num_eigvals])

        # Global distance statistics
        tri_mask = torch.triu(mask_2d, diagonal=1)
        tri_sum = tri_mask.sum(dim=(1, 2)).clamp_min(1)
        mean_dist = (D * tri_mask).sum(dim=(1, 2)) / tri_sum
        max_dist = (D * tri_mask).amax(dim=(1, 2))
        dist_var = ((D - mean_dist[:, None, None]).square() * tri_mask).sum(dim=(1, 2)) / tri_sum
        compactness = mean_dist / (max_dist + 1e-6)

        features.append(torch.stack([mean_dist, max_dist, dist_var, compactness], dim=-1))

        raw_features = torch.cat(features, dim=-1)  # stays float32
        return self.spectral_proj(raw_features).to(point_cloud.dtype)
