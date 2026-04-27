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


class HodgeTopologyBranch(nn.Module):
    NUM_SPECTRAL_SCALES = 4
    NUM_SPECTRAL_EIGVALS = 4
    MAX_K = 16  # Fixed maximum simplex size for combinatorial limits

    def __init__(self, lift_dim: int, summary_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.summary_dim = summary_dim
        self.surrogate_proj = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Differentiable spectral geometry features now include BOTH beta_0 and beta_1
        # so the dimension is NUM_SCALES * NUM_EIGS * 2 + 4
        spectral_feature_dim = self.NUM_SPECTRAL_SCALES * self.NUM_SPECTRAL_EIGVALS * 2 + 4
        self.log_scales = nn.Parameter(torch.linspace(-1.5, 1.5, self.NUM_SPECTRAL_SCALES))
        
        self.spectral_proj = nn.Sequential(
            nn.Linear(spectral_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lift_dim = lift_dim

        # Precompute Combinatorial Boundary Matrices for the Simplicial Complex
        self._build_boundaries(self.MAX_K)

    def _build_boundaries(self, K: int) -> None:
        """Precomputes B1 (Vertices to Edges) and B2 (Edges to Triangles) operators."""
        edges = []
        for i in range(K):
            for j in range(i+1, K):
                edges.append((i, j))
                
        triangles = []
        for i in range(K):
            for j in range(i+1, K):
                for k in range(j+1, K):
                    triangles.append((i, j, k))
                    
        B1 = torch.zeros(K, len(edges))
        for e_idx, (i, j) in enumerate(edges):
            B1[i, e_idx] = -1
            B1[j, e_idx] = 1
            
        B2 = torch.zeros(len(edges), len(triangles))
        edge_to_idx = {e: idx for idx, e in enumerate(edges)}
        for t_idx, (i, j, k) in enumerate(triangles):
            B2[edge_to_idx[(j, k)], t_idx] = 1
            B2[edge_to_idx[(i, k)], t_idx] = -1
            B2[edge_to_idx[(i, j)], t_idx] = 1
            
        self.register_buffer("B1", B1)
        self.register_buffer("B2", B2)
        
        # Save indices to rapidly build weight matrices during forward pass
        self.e_idx_i = [e[0] for e in edges]
        self.e_idx_j = [e[1] for e in edges]
        self.t_idx_ij = [edge_to_idx[(t[0], t[1])] for t in triangles]
        self.t_idx_jk = [edge_to_idx[(t[1], t[2])] for t in triangles]
        self.t_idx_ik = [edge_to_idx[(t[0], t[2])] for t in triangles]

    def _surrogate_summary(self, lifted_tokens: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
        def _safe_norm(x: torch.Tensor, dim: int) -> torch.Tensor:
            return torch.sqrt(x.square().sum(dim=dim) + 1e-6)

        B, N, k = lifted_tokens.shape
        K_eff = min(N, self.MAX_K)
        _, top_idx = torch.topk(activations, K_eff, dim=1, sorted=False)
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, k)
        tokens = torch.gather(lifted_tokens, 1, top_idx_exp)
        act = torch.gather(activations, 1, top_idx)

        weights = act.unsqueeze(-1)
        weighted = tokens * weights
        mass = act.sum(dim=1, keepdim=True)
        denom = mass.clamp_min(1.0)
        centroid = weighted.sum(dim=1) / denom
        diffs = tokens - centroid.unsqueeze(1)

        pairwise = torch.cdist(tokens, tokens)
        pair_weights = torch.matmul(act.unsqueeze(-1), act.unsqueeze(1))
        tri_mask = torch.triu(torch.ones(K_eff, K_eff, device=tokens.device), diagonal=1).unsqueeze(0)
        tri_weights = pair_weights * tri_mask
        weighted_pairwise = pairwise * tri_weights
        pair_mass = tri_weights.sum(dim=(1, 2)).clamp_min(1.0)

        mean_pair = weighted_pairwise.sum(dim=(1, 2)) / pair_mass
        max_pair = weighted_pairwise.amax(dim=(1, 2))
        pair_centered = (pairwise - mean_pair[:, None, None]) * tri_weights
        pair_var = (pair_centered.square().sum(dim=(1, 2)) / pair_mass).clamp_min(0.0)
        pair_std = torch.sqrt(pair_var + 1e-6)

        disp = _safe_norm(diffs, dim=-1)
        weighted_disp = disp * act
        mean_disp = weighted_disp.sum(dim=1) / denom.squeeze(1)
        max_disp = weighted_disp.amax(dim=1)
        disp_centered = (disp - mean_disp[:, None]) * act
        disp_var = (disp_centered.square().sum(dim=1) / denom.squeeze(1)).clamp_min(0.0)
        disp_std = torch.sqrt(disp_var + 1e-6)

        support_ratio = (act > 1e-3).float().mean(dim=1)
        activation_mean = act.mean(dim=1)
        activation_std = act.std(dim=1, unbiased=False)
        centroid_norm = _safe_norm(centroid, dim=1)
        token_norm_mean = (_safe_norm(tokens, dim=-1) * act).sum(dim=1) / denom.squeeze(1)
        second_moment = torch.sqrt(weighted.square().sum(dim=(1, 2)) / denom.squeeze(1) + 1e-6)

        return torch.stack([
            mean_pair, max_pair, pair_std, mean_disp, max_disp, disp_std,
            support_ratio, activation_mean, activation_std, centroid_norm,
            token_norm_mean, second_moment
        ], dim=-1)

    def surrogate(self, lifted_tokens: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
        summary = self._surrogate_summary(lifted_tokens, activations)
        return self.surrogate_proj(summary)

    def exact(self, point_cloud: np.ndarray, Q: int, max_edge_length: float | None = None) -> tuple[List[Any], np.ndarray]:
        diagrams = compute_persistence_diagrams(point_cloud, Q, max_edge_length=max_edge_length)
        return diagrams, summarize_diagrams(diagrams)

    def spectral_features(self, point_cloud: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
        """Compute fully differentiable Hodge Laplacian features for beta_0 and beta_1."""
        B_batch, N, k_dim = point_cloud.shape

        cloud_f32 = point_cloud.float()
        act_f32 = activations.float()

        # Gather top K_eff points
        K_eff = min(N, self.MAX_K)
        _, top_idx = torch.topk(act_f32, K_eff, dim=1, sorted=False)
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, k_dim)
        cloud_c = torch.gather(cloud_f32, 1, top_idx_exp)
        act_c = torch.gather(act_f32, 1, top_idx)

        # Zero-pad to MAX_K so combinatorial matrices align statically
        if K_eff < self.MAX_K:
            pad_size = self.MAX_K - K_eff
            cloud_pad = torch.zeros(B_batch, pad_size, k_dim, device=cloud_c.device)
            act_pad = torch.zeros(B_batch, pad_size, device=act_c.device)
            cloud_c = torch.cat([cloud_c, cloud_pad], dim=1)
            act_c = torch.cat([act_c, act_pad], dim=1)

        mask = (act_c > 1e-3).float()
        mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
        D = torch.cdist(cloud_c, cloud_c) * mask_2d

        scales = torch.exp(self.log_scales.float())
        num_eigvals = self.NUM_SPECTRAL_EIGVALS
        
        # W0: Vertex weights (B, K)
        W0 = act_c
        
        features = []
        for s_idx in range(self.NUM_SPECTRAL_SCALES):
            sigma = scales[s_idx]
            
            # W1: Edge weights (B, E)
            # Edge weights = exp(-d^2 / 2sigma^2) * (act_i * act_j)
            A = torch.exp(-D.square() / (2 * sigma.square() + 1e-8)) * mask_2d
            W1 = A[:, self.e_idx_i, self.e_idx_j]
            
            # W2: Triangle weights (B, T)
            # Simplest valid filtration weight: product of the 3 edges
            W2 = W1[:, self.t_idx_ij] * W1[:, self.t_idx_jk] * W1[:, self.t_idx_ik]
            
            # Form L0 = B1 W1 B1^T
            # B1 is (K, E). W1 is (B, E).
            W1_diag = torch.diag_embed(W1)
            L0 = torch.matmul(self.B1, torch.matmul(W1_diag, self.B1.t()))
            L0 = L0 + torch.eye(self.MAX_K, device=L0.device).unsqueeze(0) * 1e-6
            eigvals_L0 = torch.linalg.eigvalsh(L0)
            
            # Form L1 = B1^T W0 B1 + B2 W2 B2^T
            # B1^T is (E, K), W0 is (B, K). B2 is (E, T), W2 is (B, T).
            W0_diag = torch.diag_embed(W0)
            W2_diag = torch.diag_embed(W2)
            
            term1 = torch.matmul(self.B1.t(), torch.matmul(W0_diag, self.B1))
            term2 = torch.matmul(self.B2, torch.matmul(W2_diag, self.B2.t()))
            L1 = term1 + term2
            L1 = L1 + torch.eye(len(self.e_idx_i), device=L1.device).unsqueeze(0) * 1e-6
            eigvals_L1 = torch.linalg.eigvalsh(L1)
            
            features.append(eigvals_L0[:, :num_eigvals])
            features.append(eigvals_L1[:, :num_eigvals])

        # Global distance statistics
        tri_mask = torch.triu(mask_2d, diagonal=1)
        tri_sum = tri_mask.sum(dim=(1, 2)).clamp_min(1)
        mean_dist = (D * tri_mask).sum(dim=(1, 2)) / tri_sum
        max_dist = (D * tri_mask).amax(dim=(1, 2))
        dist_var = ((D - mean_dist[:, None, None]).square() * tri_mask).sum(dim=(1, 2)) / tri_sum
        compactness = mean_dist / (max_dist + 1e-6)

        features.append(torch.stack([mean_dist, max_dist, dist_var, compactness], dim=-1))

        raw_features = torch.cat(features, dim=-1)
        return self.spectral_proj(raw_features).to(point_cloud.dtype)
