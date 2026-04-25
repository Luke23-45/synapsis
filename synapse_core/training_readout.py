"""Training-time relaxed readout helpers for Z2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .anchor_selector import solve_relaxed_selector
from .event_encoder import hysteretic_event_score, sharp_event_score
from .geometric_lift import apply_lift, normalize_anchors
from .saliency_normalizer import normalize_saliency


@dataclass
class RelaxedReadoutState:
    """Continuous training-time readout derived from the relaxed selector."""

    event_scores: np.ndarray
    saliency_scores: np.ndarray
    y_star: np.ndarray
    candidate_vectors: np.ndarray
    normalized_vectors: np.ndarray
    lifted_candidates: np.ndarray
    weighted_cloud: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray


def candidate_anchor_vectors(
    trajectory: np.ndarray,
    event_scores: np.ndarray,
) -> np.ndarray:
    """
    Build fixed-size candidate vectors for every timestep.

    This is a training-time surrogate space: one candidate vector per timestep,
    with a unit local duration for all non-initial steps. It preserves dependence
    on the normalized learned lift while keeping the readout continuous in ``y``.
    """
    if trajectory.ndim != 2:
        raise ValueError(f"trajectory must have shape (T, d), got ndim={trajectory.ndim}")
    if event_scores.ndim != 1 or event_scores.shape[0] != trajectory.shape[0]:
        raise ValueError("event_scores must have shape (T,) matching trajectory")

    T, d = trajectory.shape
    vectors = np.zeros((T, d + 3), dtype=np.float64)
    vectors[:, 0] = (np.arange(T, dtype=np.float64) + 1.0) / max(T, 1)
    vectors[:, 1 : 1 + d] = trajectory
    vectors[1:, 1 + d] = 1.0
    vectors[:, 2 + d] = event_scores
    return vectors


def relaxed_weighted_cloud(
    candidate_vectors_norm: np.ndarray,
    W_Theta: np.ndarray,
    y_star: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the learned lift and weight every candidate point by ``y_star``."""
    lifted = apply_lift(candidate_vectors_norm, W_Theta)
    weighted = lifted * y_star[:, None]
    return lifted, weighted


def compute_relaxed_readout(
    trajectory: np.ndarray,
    K: int,
    r: int,
    lam: float,
    W_Theta: np.ndarray,
    *,
    alpha: float = 0.0,
    phi: Optional[Callable] = None,
    eta: Optional[Callable] = None,
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    solver: str = "osqp",
    saliency_mode: str = "identity",
    saliency_temperature: float = 1.0,
    saliency_eps: float = 1e-6,
) -> RelaxedReadoutState:
    """Compute the continuous training-time relaxed readout from §13."""
    if alpha == 0.0 and phi is None and eta is None:
        event_scores = sharp_event_score(trajectory)
    else:
        event_scores, _ = hysteretic_event_score(trajectory, alpha=alpha, phi=phi, eta=eta)

    saliency_scores = normalize_saliency(
        event_scores,
        mode=saliency_mode,
        temperature=saliency_temperature,
        eps=saliency_eps,
    )
    y_star = solve_relaxed_selector(saliency_scores, K, r, lam, solver=solver)

    candidate_vectors = candidate_anchor_vectors(trajectory, event_scores)
    normalized_vectors, mu_used, sigma_used = normalize_anchors(candidate_vectors, mu=mu, sigma=sigma)
    lifted_candidates, weighted_cloud = relaxed_weighted_cloud(normalized_vectors, W_Theta, y_star)

    return RelaxedReadoutState(
        event_scores=event_scores,
        saliency_scores=saliency_scores,
        y_star=y_star,
        candidate_vectors=candidate_vectors,
        normalized_vectors=normalized_vectors,
        lifted_candidates=lifted_candidates,
        weighted_cloud=weighted_cloud,
        mu=mu_used,
        sigma=sigma_used,
    )
