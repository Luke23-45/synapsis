"""
Baseline feature extraction and lightweight model definitions — Z2 version.

Z2 Reference: §5–9, §11, §13 of 02_rigorous_architecture.md
Replaces Z1 anchor_feature/synapse_feature (tau, weights) with Z2 pipeline
(relaxed selector → hard projection → normalize → lift).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist

from synapse_core.anchor_selector import Anchor, solve_relaxed_selector, hard_projection, build_anchors
from synapse_core.event_encoder import sharp_event_score
from synapse_core.geometric_lift import anchor_vectors, normalize_anchors, apply_lift
from synapse_core.memory_operator import compute_memory
from experiments.empirical.common.math_utils import pad_rows as _pad_rows


# _pad_rows imported from experiments.empirical.common.math_utils


def summarize_diagrams(diagrams: List) -> np.ndarray:
    summary: List[float] = []
    for dgm in diagrams:
        points = np.asarray(dgm.points, dtype=np.float32) if hasattr(dgm, "points") else np.asarray(dgm, dtype=np.float32)
        if points.size == 0:
            summary.extend([0.0, 0.0, 0.0, 0.0])
            continue
        finite = points[np.isfinite(points[:, 1])]
        pers = finite[:, 1] - finite[:, 0] if len(finite) else np.zeros(0, dtype=np.float32)
        
        # Filter numerical artifacts (tiny bars) which explode under Standard Scaler
        pers = pers[pers > 1e-4]
        
        summary.extend([
            float(len(pers)),
            float(pers.mean()) if len(pers) else 0.0,
            float(pers.max()) if len(pers) else 0.0,
            float(pers.sum()) if len(pers) else 0.0,
        ])
    return np.asarray(summary, dtype=np.float32)


def proxy_topology_features(sequence: np.ndarray, k: int) -> np.ndarray:
    """Proxy features approximating topological summaries via cdist."""
    if len(sequence) == 0:
        return np.zeros(8, dtype=np.float32)
    sample = sequence[np.linspace(0, len(sequence) - 1, num=min(k, len(sequence)), dtype=int)]
    if len(sample) < 2:
        return np.zeros(8, dtype=np.float32)
    # Use cdist (C-optimised) instead of O(n²) numpy broadcast
    dists = cdist(sample, sample, metric="euclidean")
    tri = dists[np.triu_indices_from(dists, k=1)]
    velocity = np.linalg.norm(np.diff(sample, axis=0), axis=1) if len(sample) > 1 else np.zeros(1, dtype=np.float32)
    return np.asarray(
        [
            float(tri.mean()) if len(tri) else 0.0,
            float(tri.std()) if len(tri) else 0.0,
            float(tri.max()) if len(tri) else 0.0,
            float(np.percentile(tri, 75)) if len(tri) else 0.0,
            float(velocity.mean()),
            float(velocity.std()),
            float(np.max(velocity)),
            float(np.percentile(velocity, 75)),
        ],
        dtype=np.float32,
    )


def uniform_feature(sequence: np.ndarray, k: int) -> np.ndarray:
    if len(sequence) == 0:
        return np.zeros((k, 1), dtype=np.float32).reshape(-1)
    idx = np.linspace(0, len(sequence) - 1, num=min(k, len(sequence)), dtype=int)
    return _pad_rows(sequence[idx], k).reshape(-1)


# =======================================================================
# Z2 Feature Extraction Functions
# =======================================================================

def anchor_feature_z2(
    sequence: np.ndarray,
    K: int,
    r: int,
    lam: float,
    W_Theta: np.ndarray,
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Z2 anchor feature: relaxed selector → hard projection → normalize → lift."""
    scores = sharp_event_score(sequence)
    y_star = solve_relaxed_selector(scores, K, r, lam, solver="osqp")
    indices = hard_projection(y_star, K, r)
    anchors = build_anchors(indices, sequence, scores)
    V = anchor_vectors(anchors)
    V_norm, _, _ = normalize_anchors(V, mu=mu, sigma=sigma)
    cloud = apply_lift(V_norm, W_Theta)
    cloud = cloud.astype(np.float32) if cloud.size else np.zeros((0, W_Theta.shape[0]), dtype=np.float32)
    return _pad_rows(cloud, K).reshape(-1), indices


def synapse_feature_z2(
    sequence: np.ndarray,
    K: int,
    r: int,
    lam: float,
    k: int,
    Q: int,
    W_Theta: np.ndarray,
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[int], int]:
    """Z2 SYNAPSE feature: full pipeline with topology summary."""
    state = compute_memory(sequence, K=K, r=r, lam=lam, W_Theta=W_Theta, Q=Q,
                           mu=mu, sigma=sigma, solver="osqp")
    cloud = state.point_cloud.astype(np.float32) if state.point_cloud.size else np.zeros((0, k), dtype=np.float32)
    base = _pad_rows(cloud, K).reshape(-1)
    topo = summarize_diagrams(state.persistence_diagrams)
    feature = np.concatenate([base, topo], axis=0).astype(np.float32)
    memory_size = int(cloud.size + topo.size)
    return feature, state.anchor_indices, memory_size


def relaxed_anchor_feature(
    sequence: np.ndarray,
    K: int,
    r: int,
    lam: float,
    k: int,
    W_Theta: np.ndarray,
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[int], int]:
    """Z2 relaxed readout: uses y* weights directly (training-time feature, §13)."""
    scores = sharp_event_score(sequence)
    y_star = solve_relaxed_selector(scores, K, r, lam, solver="osqp")
    # Weighted combination of all anchors (not just hard-projected subset)
    indices = [t for t in range(len(y_star)) if y_star[t] > 0.01]
    anchors = build_anchors(indices, sequence, scores)
    V = anchor_vectors(anchors)
    V_norm, _, _ = normalize_anchors(V, mu=mu, sigma=sigma)
    cloud = apply_lift(V_norm, W_Theta)
    # Weight each lifted point by its y* value
    weights = y_star[indices]
    weighted_cloud = cloud * weights[:, None]
    cloud_out = weighted_cloud.astype(np.float32) if weighted_cloud.size else np.zeros((0, k), dtype=np.float32)
    base = _pad_rows(cloud_out, K).reshape(-1)
    memory_size = int(cloud_out.size)
    return base, indices, memory_size


# =======================================================================
# Model Heads (reusable from Z1)
# =======================================================================

class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRUHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1])


class TransformerHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, max_len: int = 512, num_heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) + self.pos[:, : x.shape[1], :]
        encoded = self.encoder(x)
        return self.head(encoded[:, -1, :])


def build_baseline(name: str, input_dim: int, output_dim: int, hidden_dim: int, max_len: int = 512) -> nn.Module:
    if name in {"B0", "B3", "B4", "B5", "B6", "B7"}:
        return MLPHead(input_dim, hidden_dim, output_dim)
    if name == "B1":
        return GRUHead(input_dim, hidden_dim, output_dim)
    if name == "B2":
        return TransformerHead(input_dim, hidden_dim, output_dim, max_len=max_len)
    raise KeyError(f"Unknown baseline: {name}")
