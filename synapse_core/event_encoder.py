"""
Event Encoder — Formal Math Reference: §3–4 of 02_rigorous_architecture.md

Implements two event scoring modes:

1. Sharp special case (α=0):
       e_t = ‖x_t − x_{t−1}‖₂,  t ≥ 2;  e_1 = 0.
   Reference: §3 Canonical sharp special case

2. Hysteretic event encoder (general):
       h_1 = 0
       h_t = α_t · h_{t−1} + (1 − α_t) · φ(x_t, x_{t−1}),  t ≥ 2
       e_1 = 0
       e_t = η(h_t, x_t, x_{t−1}),  t ≥ 2
   Reference: §3 Hysteretic Event Encoder

Both modes preserve causality: e_t depends only on x_{1:t}.
"""

import numpy as np
from typing import Callable, Optional, Union


def sharp_event_score(trajectory: np.ndarray) -> np.ndarray:
    """
    Compute the sharp (non-hysteretic) event score.

    Formal definition (§3, canonical sharp special case):
        e_1 = 0
        e_t = ‖x_t − x_{t−1}‖₂,  t ≥ 2

    Parameters
    ----------
    trajectory : np.ndarray, shape (T, d)
        The input trajectory x_{1:T}.

    Returns
    -------
    scores : np.ndarray, shape (T,)
        Event scores e_1, ..., e_T.
    """
    T = trajectory.shape[0]
    if T < 1:
        raise ValueError("Trajectory must have at least 1 timestep.")

    scores = np.zeros(T, dtype=np.float64)
    if T >= 2:
        # e_t = ‖x_t − x_{t−1}‖₂ for t ≥ 2
        diffs = trajectory[1:] - trajectory[:-1]  # shape (T-1, d)
        scores[1:] = np.linalg.norm(diffs, axis=1)

    return scores


def hysteretic_event_score(
    trajectory: np.ndarray,
    alpha: Union[float, np.ndarray, Callable],
    phi: Optional[Callable] = None,
    eta: Optional[Callable] = None,
) -> tuple:
    """
    Compute hysteretic event scores with latent state recursion.

    Formal definition (§3 of 02_rigorous_architecture.md):
        h_1 = 0  ∈ ℝ^p
        h_t = α_t · h_{t−1} + (1 − α_t) · φ(x_t, x_{t−1}),  t ≥ 2
        e_1 = 0
        e_t = η(h_t, x_t, x_{t−1}),  t ≥ 2

    Parameters
    ----------
    trajectory : np.ndarray, shape (T, d)
        The input trajectory x_{1:T}.
    alpha : float, ndarray of shape (T,), or callable(t, x_{1:t}) -> float
        The causal gate α_t ∈ [0, 1].
        - If float: constant α for all t.
        - If ndarray: alpha[t] is used at time t.
        - If callable: alpha(t, trajectory[:t+1]) is called.
    phi : callable(x_t, x_{t-1}) -> ndarray of shape (p,), optional
        The feature map. Default: φ(x_t, x_{t-1}) = x_t − x_{t-1}.
    eta : callable(h_t, x_t, x_{t-1}) -> float, optional
        The scoring function. Default: η(h_t, x_t, x_{t-1}) = ‖h_t‖₂.

    Returns
    -------
    scores : np.ndarray, shape (T,)
        Event scores e_1, ..., e_T.
    latent_states : np.ndarray, shape (T, p)
        Latent states h_1, ..., h_T.
    """
    T, d = trajectory.shape
    if T < 1:
        raise ValueError("Trajectory must have at least 1 timestep.")

    # Default φ: difference
    if phi is None:
        def phi(x_t, x_prev):
            return x_t - x_prev

    # Default η: L2 norm of latent state
    if eta is None:
        def eta(h_t, x_t, x_prev):
            return np.linalg.norm(h_t)

    # Determine latent dimension p from a probe call
    if T >= 2:
        probe = phi(trajectory[1], trajectory[0])
        p = probe.shape[0] if isinstance(probe, np.ndarray) else 1
    else:
        p = d  # fallback

    latent_states = np.zeros((T, p), dtype=np.float64)
    scores = np.zeros(T, dtype=np.float64)

    # h_1 = 0, e_1 = 0 (already initialized)

    for t in range(1, T):
        # Resolve α_t
        if callable(alpha):
            alpha_t = alpha(t, trajectory[: t + 1])
        elif isinstance(alpha, np.ndarray):
            alpha_t = alpha[t]
        else:
            alpha_t = float(alpha)

        # Validate α_t ∈ [0, 1]
        if not (0.0 <= alpha_t <= 1.0):
            raise ValueError(f"alpha_t={alpha_t} at t={t} is not in [0, 1].")

        # h_t = α_t · h_{t−1} + (1 − α_t) · φ(x_t, x_{t−1})
        phi_val = phi(trajectory[t], trajectory[t - 1])
        if not isinstance(phi_val, np.ndarray):
            phi_val = np.array([phi_val], dtype=np.float64)

        latent_states[t] = alpha_t * latent_states[t - 1] + (1.0 - alpha_t) * phi_val

        # e_t = η(h_t, x_t, x_{t−1})
        scores[t] = eta(latent_states[t], trajectory[t], trajectory[t - 1])

    return scores, latent_states
