"""
Stress transforms for paper-claims experiments.
"""

from __future__ import annotations

import numpy as np


def add_observation_noise(sequence: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    return (sequence + rng.normal(scale=sigma, size=sequence.shape)).astype(np.float32)


def truncate_history(sequence: np.ndarray, keep: int) -> np.ndarray:
    if keep >= len(sequence):
        return sequence.copy()
    return sequence[-keep:].copy()


def inject_distractors(sequence: np.ndarray, rng: np.random.Generator, density: float, scale: float) -> np.ndarray:
    out = sequence.copy()
    mask = rng.random(len(out)) < density
    out[mask] += rng.normal(scale=scale, size=out[mask].shape)
    return out.astype(np.float32)


def random_drop_nonanchors(
    sequence: np.ndarray,
    anchor_indices: list[int],
    rng: np.random.Generator,
    keep_prob: float,
) -> np.ndarray:
    keep = np.zeros(len(sequence), dtype=bool)
    keep[anchor_indices] = True
    keep |= rng.random(len(sequence)) < keep_prob
    if not np.any(keep):
        keep[-1] = True
    return sequence[keep].astype(np.float32)

