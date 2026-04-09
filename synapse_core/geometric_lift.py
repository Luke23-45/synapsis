"""
Geometric Lift — Formal Math Reference: §6 of 01_main_definition.md, §5 of 02_rigorous_architecture.md

Implements the weighted geometric embedding that lifts anchors into ℝ^{d+3}:

    ρ(a_j) = (√w_t · t_j,  √w_x · s_j,  √w_δ · δ_j,  √w_e · ξ_j)  ∈ ℝ^{d+3}

The lifted anchor cloud is:
    P(x_{1:T}) = {ρ(a_1), ..., ρ(a_m)}

with Euclidean metric d_P(u,v) = ‖u − v‖₂.

If m = 0, then P = ∅.
"""

import numpy as np
from typing import List, Tuple
from .anchor_selector import Anchor


def lift_single_anchor(
    anchor: Anchor,
    weights: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    Lift a single anchor into the weighted embedding space.

    Formal definition (§6 of 01_main_definition.md):
        ρ(a_j) = (√w_t · t_j,  √w_x · s_j,  √w_δ · δ_j,  √w_e · ξ_j)

    Parameters
    ----------
    anchor : Anchor
        The anchor (t_j, s_j, δ_j, ξ_j).
    weights : tuple of 4 positive floats
        (w_t, w_x, w_δ, w_e) — all must be > 0.

    Returns
    -------
    point : np.ndarray, shape (d+3,)
        The lifted point ρ(a_j) in ℝ^{d+3}.
    """
    w_t, w_x, w_delta, w_e = weights

    if w_t <= 0 or w_x <= 0 or w_delta <= 0 or w_e <= 0:
        raise ValueError(f"All weights must be strictly positive. Got: {weights}")

    sqrt_wt = np.sqrt(w_t)
    sqrt_wx = np.sqrt(w_x)
    sqrt_wd = np.sqrt(w_delta)
    sqrt_we = np.sqrt(w_e)

    # s_j may be a vector of dimension d
    s_j = anchor.s
    d = s_j.shape[0]

    # Construct ρ(a_j) ∈ ℝ^{d+3}
    # Components: [√w_t · t_j, √w_x · s_j[0], ..., √w_x · s_j[d-1], √w_δ · δ_j, √w_e · ξ_j]
    point = np.zeros(d + 3, dtype=np.float64)
    point[0] = sqrt_wt * anchor.t
    point[1 : d + 1] = sqrt_wx * s_j
    point[d + 1] = sqrt_wd * anchor.delta
    point[d + 2] = sqrt_we * anchor.xi

    return point


def lift_anchors(
    anchors: List[Anchor],
    weights: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    Lift all anchors into the weighted embedding space to form the point cloud.

    Formal definition (§6 of 01_main_definition.md):
        P(x_{1:T}) = {ρ(a_1), ..., ρ(a_m)}

    If m = 0 (no anchors), returns an empty array with shape (0, 0).

    Parameters
    ----------
    anchors : list of Anchor
        The anchor sequence A(x_{1:T}).
    weights : tuple of 4 positive floats
        (w_t, w_x, w_δ, w_e).

    Returns
    -------
    cloud : np.ndarray, shape (m, d+3) or (0, 0) if m=0
        The lifted anchor cloud P(x_{1:T}).
    """
    if len(anchors) == 0:
        return np.empty((0, 0), dtype=np.float64)

    points = [lift_single_anchor(a, weights) for a in anchors]
    return np.stack(points, axis=0)
