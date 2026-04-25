"""Causal saliency normalization for the Z2 selector input."""

from __future__ import annotations

import numpy as np


def causal_running_stats(values: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Return causal running means and standard deviations."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {values.shape}")

    steps = np.arange(1, values.size + 1, dtype=np.float64)
    csum = np.cumsum(values, dtype=np.float64)
    csum_sq = np.cumsum(values * values, dtype=np.float64)

    mean = csum / steps
    var = np.maximum(csum_sq / steps - mean * mean, 0.0)
    std = np.sqrt(var + eps)
    return mean, std


def normalize_saliency(
    event_scores: np.ndarray,
    mode: str = "identity",
    temperature: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Apply a causal saliency normalizer to event scores.

    Modes
    -----
    ``identity``:
        ``s_t = e_t``
    ``z_score``:
        causal running z-score standardization
    ``temperature``:
        causal z-score followed by logistic confidence shaping and rescaling by ``e_t``
    """
    event_scores = np.asarray(event_scores, dtype=np.float64)
    if event_scores.ndim != 1:
        raise ValueError(f"event_scores must be one-dimensional, got shape {event_scores.shape}")
    if event_scores.size == 0:
        raise ValueError("event_scores must be non-empty")
    if np.any(event_scores < 0):
        raise ValueError("event_scores must be non-negative")
    if mode not in {"identity", "z_score", "temperature"}:
        raise ValueError(f"unsupported saliency mode: {mode!r}")
    if mode == "temperature" and temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    saliency = np.array(event_scores, copy=True)
    if mode == "identity":
        saliency[0] = 0.0
        return saliency

    mean, std = causal_running_stats(event_scores, eps=eps)
    z = (event_scores - mean) / std

    if mode == "z_score":
        saliency = z
    else:
        saliency = event_scores * (1.0 / (1.0 + np.exp(-z / temperature)))

    saliency[0] = 0.0
    return saliency
