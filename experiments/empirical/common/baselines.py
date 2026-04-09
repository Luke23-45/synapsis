"""
Baseline feature extraction and lightweight model definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn

from synapse_core.anchor_selector import Anchor, select_anchors
from synapse_core.event_encoder import sharp_event_score
from synapse_core.geometric_lift import lift_anchors
from synapse_core.memory_operator import M


def _pad_rows(arr: np.ndarray, rows: int) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0:
        cols = arr.shape[1] if arr.ndim == 2 else 1
        return np.zeros((rows, cols), dtype=np.float32)
    if arr.shape[0] >= rows:
        return arr[:rows].astype(np.float32)
    pad = np.zeros((rows - arr.shape[0], arr.shape[1]), dtype=np.float32)
    return np.concatenate([arr.astype(np.float32), pad], axis=0)


def summarize_diagrams(diagrams: List) -> np.ndarray:
    summary: List[float] = []
    for dgm in diagrams:
        points = np.asarray(dgm.points, dtype=np.float32) if hasattr(dgm, "points") else np.asarray(dgm, dtype=np.float32)
        if points.size == 0:
            summary.extend([0.0, 0.0, 0.0, 0.0])
            continue
        finite = points[np.isfinite(points[:, 1])]
        pers = finite[:, 1] - finite[:, 0] if len(finite) else np.zeros(0, dtype=np.float32)
        summary.extend([
            float(len(finite)),
            float(pers.mean()) if len(pers) else 0.0,
            float(pers.max()) if len(pers) else 0.0,
            float(pers.sum()) if len(pers) else 0.0,
        ])
    return np.asarray(summary, dtype=np.float32)


def proxy_topology_features(sequence: np.ndarray, k: int) -> np.ndarray:
    if len(sequence) == 0:
        return np.zeros(8, dtype=np.float32)
    sample = sequence[np.linspace(0, len(sequence) - 1, num=min(k, len(sequence)), dtype=int)]
    if len(sample) < 2:
        return np.zeros(8, dtype=np.float32)
    dists = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
    tri = dists[np.triu_indices_from(dists, k=1)]
    velocity = np.linalg.norm(np.diff(sequence, axis=0), axis=1) if len(sequence) > 1 else np.zeros(1, dtype=np.float32)
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


def anchor_feature(sequence: np.ndarray, k: int, r: int, tau: float, weights: Tuple[float, float, float, float]) -> Tuple[np.ndarray, List[int]]:
    scores = sharp_event_score(sequence)
    indices, anchors = select_anchors(scores, sequence, k, r, tau)
    cloud = lift_anchors(anchors, weights)
    cloud = cloud.astype(np.float32) if cloud.size else np.zeros((0, sequence.shape[1] + 3), dtype=np.float32)
    return _pad_rows(cloud, k).reshape(-1), indices


def delta_anchor_feature(sequence: np.ndarray, k: int, r: int, tau: float, weights: Tuple[float, float, float, float]) -> Tuple[np.ndarray, List[int]]:
    diffs = np.zeros(len(sequence), dtype=np.float32)
    if len(sequence) > 1:
        diffs[1:] = np.linalg.norm(np.diff(sequence, axis=0), axis=1)
    indices, anchors = select_anchors(diffs, sequence, k, r, tau)
    cloud = lift_anchors(anchors, weights)
    cloud = cloud.astype(np.float32) if cloud.size else np.zeros((0, sequence.shape[1] + 3), dtype=np.float32)
    return _pad_rows(cloud, k).reshape(-1), indices


def synapse_feature(
    sequence: np.ndarray,
    k: int,
    r: int,
    tau: float,
    q: int,
    weights: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, List[int], int]:
    state = M(sequence, K=k, r=r, tau=tau, weights=weights, Q=q)
    cloud = state.point_cloud.astype(np.float32) if state.point_cloud.size else np.zeros((0, sequence.shape[1] + 3), dtype=np.float32)
    base = _pad_rows(cloud, k).reshape(-1)
    topo = summarize_diagrams(state.persistence_diagrams)
    feature = np.concatenate([base, topo], axis=0).astype(np.float32)
    memory_size = int(cloud.size + topo.size)
    return feature, state.anchor_indices, memory_size


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
    if name in {"B0", "B3", "B4", "B5", "B6"}:
        return MLPHead(input_dim, hidden_dim, output_dim)
    if name == "B1":
        return GRUHead(input_dim, hidden_dim, output_dim)
    if name == "B2":
        return TransformerHead(input_dim, hidden_dim, output_dim, max_len=max_len)
    raise KeyError(f"Unknown baseline: {name}")

