"""
Geometric Lift — Z2 Reference: §8–9 of 02_rigorous_architecture.md

Z2 Lift Pipeline:
    1. Build anchor vectors:  v(a_j) = [t_j, s_jᵀ, δ_j, ξ_j]ᵀ ∈ ℝ^{d+3}  [§7–8]
    2. Normalize:            N(v) = D⁻¹(v − μ)                          [§8]
    3. Learned lift:         ρ_Θ(a_j) = W_Θ N(v(a_j)) ∈ ℝ^k           [§9]

Legacy Z1 lift (lift_anchors with fixed diagonal weights) retained for
backward compatibility.
"""

import numpy as np
from typing import List, Tuple, Optional
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


# =======================================================================
# Z2 Anchor Vector Construction — §7–8 of 02_rigorous_architecture.md
# =======================================================================

def anchor_vectors(
    anchors: List[Anchor],
    D: Optional[int] = None,
) -> np.ndarray:
    """
    Build the anchor vector matrix V from the anchor sequence A*.

    Z2 Reference: §7–8 of 02_rigorous_architecture.md

    For each anchor a_j = (t_j, s_j, δ_j, ξ_j):
        v(a_j) = [t_j, s_jᵀ, δ_j, ξ_j]ᵀ ∈ ℝ^{d+3}

    Parameters
    ----------
    anchors : list of Anchor
        The anchor sequence A*.

    Returns
    -------
    V : np.ndarray, shape (m, d+3)
        Matrix of anchor vectors, one per row.
        Returns empty array of shape (0, d+3) if anchors is empty.
    """
    if not anchors:
        if D is None:
            return np.empty((0, 0), dtype=np.float64)
        return np.empty((0, D), dtype=np.float64)

    m = len(anchors)
    d = anchors[0].s.shape[0]  # state dimension
    D = d + 3
    V = np.zeros((m, D), dtype=np.float64)

    for j, a in enumerate(anchors):
        V[j, 0] = a.t                    # t_j (continuous)
        V[j, 1:1 + d] = a.s              # s_j (state, d-dimensional)
        V[j, 1 + d] = a.delta            # δ_j (discrete timesteps)
        V[j, 2 + d] = a.xi               # ξ_j (event intensity)

    return V


# =======================================================================
# Z2 Normalization — §8 of 02_rigorous_architecture.md
# =======================================================================

def normalize_anchors(
    V: np.ndarray,
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Z2 affine normalization N(v) = D⁻¹(v − μ) to anchor vectors.

    Z2 Reference: §8 of 02_rigorous_architecture.md
    Formal Claims: N is bijective when σ_ℓ > 0 for all ℓ

    Parameters
    ----------
    V : np.ndarray, shape (m, D) where D = d+3
        Matrix of anchor vectors.
    mu : np.ndarray, shape (D,), optional
        Center vector. If None, computed as column-wise mean of V.
        For empty V, defaults to zero.
    sigma : np.ndarray, shape (D,), optional
        Scale vector. If None, computed as column-wise std of V.
        For empty V or constant columns, defaults to 1.0.

    Returns
    -------
    V_norm : np.ndarray, shape (m, D)
        Normalized anchor vectors N(v(a_j)).
        Returns empty array of shape (0, D) if V is empty.
    mu : np.ndarray, shape (D,)
        The center vector used.
    sigma : np.ndarray, shape (D,)
        The scale vector used. All entries are > 0 (invariant §8).
    """
    if V.shape[0] == 0:
        D = V.shape[1] if V.ndim == 2 and V.shape[1] > 0 else 0
        empty_norm = np.empty((0, D), dtype=np.float64)
        mu_out = np.zeros(D, dtype=np.float64) if D > 0 else np.array([], dtype=np.float64)
        sigma_out = np.ones(D, dtype=np.float64) if D > 0 else np.array([], dtype=np.float64)
        return empty_norm, mu_out, sigma_out

    if not np.all(np.isfinite(V)):
        raise ValueError("V must contain only finite values")

    D = V.shape[1]

    # Compute mu: column-wise mean
    if mu is None:
        mu = np.mean(V, axis=0)
    else:
        mu = np.asarray(mu, dtype=np.float64)
        if mu.shape != (D,):
            raise ValueError(f"mu shape {mu.shape} != expected ({D},)")
        if not np.all(np.isfinite(mu)):
            raise ValueError("mu must contain only finite values")

    # Compute sigma: column-wise std, enforcing σ_ℓ > 0 (§8 invariant)
    if sigma is None:
        sigma = np.std(V, axis=0)
        # Replace zero/near-zero σ with 1.0 to maintain bijectivity
        sigma[sigma <= 0] = 1.0
    else:
        sigma = np.asarray(sigma, dtype=np.float64)
        if sigma.shape != (D,):
            raise ValueError(f"sigma shape {sigma.shape} != expected ({D},)")
        if not np.all(np.isfinite(sigma)):
            raise ValueError("sigma must contain only finite values")

    # Enforce invariant: σ_ℓ > 0 for all ℓ
    if np.any(sigma <= 0):
        zero_dims = np.where(sigma <= 0)[0]
        raise ValueError(
            f"σ_ℓ must be > 0 for all ℓ (§8 bijectivity invariant). "
            f"Zero/negative at dimensions: {zero_dims.tolist()}"
        )

    # Apply N(v) = D⁻¹(v − μ)
    D_inv = np.diag(1.0 / sigma)
    V_norm = (V - mu) @ D_inv

    return V_norm, mu, sigma


# =======================================================================
# Z2 Learned Lift — §9 of 02_rigorous_architecture.md
# =======================================================================

def apply_lift(
    V_norm: np.ndarray,
    W_Theta: np.ndarray,
) -> np.ndarray:
    """
    Apply the Z2 learned lift ρ_Θ to normalized anchor vectors.

    Z2 Reference: §9 of 02_rigorous_architecture.md
    Formal Claims: ρ_Θ(a_j) = W_Θ N(v(a_j)) ∈ ℝ^k

    Parameters
    ----------
    V_norm : np.ndarray, shape (m, D) where D = d+3
        Normalized anchor vectors N(v(a_j)).
    W_Theta : np.ndarray, shape (k, D) where D = d+3
        Learned lift matrix W_Θ ∈ ℝ^{k×(d+3)}.

    Returns
    -------
    P_Theta : np.ndarray, shape (m, k)
        Lifted point cloud P_Θ = {ρ_Θ(a_j)}_{j=1}^m.
        Returns empty array of shape (0, k) if V_norm is empty.
    """
    if V_norm.shape[0] == 0:
        k = W_Theta.shape[0]
        return np.empty((0, k), dtype=np.float64)

    # P_Θ = V_norm @ W_Thetaᵀ  (each row is ρ_Θ(a_j) ∈ ℝ^k)
    P_Theta = (V_norm @ W_Theta.T).astype(np.float64)

    return P_Theta
