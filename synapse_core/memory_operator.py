"""
Memory Operator — Z2 Reference: §11 of 02_rigorous_architecture.md

Z2 Memory Operator:
    M^inf_Θ(x_{1:T}) = (A*, P_Θ, Dgm_0, ..., Dgm_Q)

Z2 Pipeline (deploy mode, §11):
    1. Event scoring (causal)          [§3–4]
    2. Relaxed selector QP             [§5]
    3. Hard projection                  [§6]
    4. Anchor construction              [§7]
    5. Normalization N(v) = D⁻¹(v−μ)   [§8]
    6. Learned lift ρ_Θ(a_j) = W_Θ N(v) [§9]
    7. Persistent topological summary    [§10]

Train mode (§13) uses relaxed readout instead of hard projection.

Legacy Z1 operator M(tau, weights) retained for backward compatibility.
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass

from .event_encoder import sharp_event_score, hysteretic_event_score
from .anchor_selector import select_anchors, Anchor, solve_relaxed_selector, hard_projection, build_anchors
from .geometric_lift import lift_anchors, anchor_vectors, normalize_anchors, apply_lift
from .saliency_normalizer import normalize_saliency
from .topological_summary import compute_persistence_diagrams, PersistenceDiagram


@dataclass
class MemoryState:
    """
    The output of the SYNAPSE memory operator M(x_{1:T}).

    Fields
    ------
    anchors : list of Anchor
        The anchor sequence A(x_{1:T}).
    anchor_indices : list of int
        The selected index set I* (0-indexed).
    point_cloud : np.ndarray, shape (m, d+3) or (0, 0)
        The lifted anchor cloud P(x_{1:T}).
    persistence_diagrams : list of PersistenceDiagram
        Q+1 persistence diagrams, Dgm_0 through Dgm_Q.
    event_scores : np.ndarray, shape (T,)
        The full event score sequence (retained for diagnostics).
    """

    anchors: List[Anchor]
    anchor_indices: List[int]
    point_cloud: np.ndarray
    persistence_diagrams: List[PersistenceDiagram]
    event_scores: np.ndarray


def M(
    trajectory: np.ndarray,
    K: int,
    r: int,
    tau: float,
    weights: Tuple[float, float, float, float],
    Q: int,
    alpha: float = 0.0,
    phi: Optional[Callable] = None,
    eta: Optional[Callable] = None,
    max_edge_length: Optional[float] = None,
) -> MemoryState:
    """
    The SYNAPSE memory operator.

    Formal definition (§8 of 01_main_definition.md):
        M(x_{1:T}) = ( A(x_{1:T}), Dgm_0(P(x_{1:T})), ..., Dgm_Q(P(x_{1:T})) )

    Parameters
    ----------
    trajectory : np.ndarray, shape (T, d)
        Input trajectory x_{1:T}.
    K : int
        Maximum anchor count.
    r : int
        Minimum refractory separation.
    tau : float
        Minimum event score threshold.
    weights : tuple of 4 positive floats
        (w_t, w_x, w_δ, w_e) for the geometric lift.
    Q : int
        Maximum homology degree.
    alpha : float, default 0.0
        Hysteresis gate. If 0.0, uses sharp event score.
    phi : callable, optional
        Feature map for hysteretic encoder.
    eta : callable, optional
        Scoring function for hysteretic encoder.
    max_edge_length : float, optional
        Maximum edge length for Rips complex.

    Returns
    -------
    state : MemoryState
        The complete memory state M(x_{1:T}).
    """
    T, d = trajectory.shape

    # Step 1: Event scoring
    if alpha == 0.0 and phi is None and eta is None:
        scores = sharp_event_score(trajectory)
    else:
        scores, _ = hysteretic_event_score(trajectory, alpha=alpha, phi=phi, eta=eta)

    # Step 2: Anchor selection
    anchor_indices, anchors = select_anchors(scores, trajectory, K, r, tau)

    # Step 3: Geometric lift
    cloud = lift_anchors(anchors, weights)

    # Step 4: Persistent homology
    diagrams = compute_persistence_diagrams(cloud, Q, max_edge_length=max_edge_length)

    return MemoryState(
        anchors=anchors,
        anchor_indices=anchor_indices,
        point_cloud=cloud,
        persistence_diagrams=diagrams,
        event_scores=scores,
    )


# =======================================================================
# Z2 Memory Operator — §11 of 02_rigorous_architecture.md
# =======================================================================

@dataclass
class Z2MemoryState:
    """
    The output of the Z2 memory operator M^inf_Θ(x_{1:T}).

    Z2 Reference: §11 of 02_rigorous_architecture.md

    Fields
    ------
    anchors : list of Anchor
        The anchor sequence A*.
    anchor_indices : list of int
        The retained index set I* (0-indexed, sorted increasing).
    y_star : np.ndarray, shape (T,)
        Relaxed selector output y* ∈ Π_{K,r,T}.
    V : np.ndarray, shape (m, d+3)
        Anchor vector matrix v(a_j).
    V_norm : np.ndarray, shape (m, d+3)
        Normalized anchor vectors N(v(a_j)).
    point_cloud : np.ndarray, shape (m, k)
        Lifted point cloud P_Θ = {ρ_Θ(a_j)}.
    persistence_diagrams : list of PersistenceDiagram
        Q+1 persistence diagrams, Dgm_0 through Dgm_Q.
    event_scores : np.ndarray, shape (T,)
        The full event score sequence.
    mu : np.ndarray, shape (d+3,)
        Center vector used in normalization.
    sigma : np.ndarray, shape (d+3,)
        Scale vector used in normalization (all > 0, §8 invariant).
    """

    anchors: List[Anchor]
    anchor_indices: List[int]
    y_star: np.ndarray
    V: np.ndarray
    V_norm: np.ndarray
    point_cloud: np.ndarray
    persistence_diagrams: List[PersistenceDiagram]
    event_scores: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray


def compute_memory(
    trajectory: np.ndarray,
    K: int,
    r: int,
    lam: float,
    W_Theta: np.ndarray,
    Q: int,
    alpha: float = 0.0,
    phi: Optional[Callable] = None,
    eta: Optional[Callable] = None,
    max_edge_length: Optional[float] = None,
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    solver: str = "osqp",
    mode: str = "deploy",
    saliency_mode: str = "identity",
    saliency_temperature: float = 1.0,
) -> Z2MemoryState:
    """
    The Z2 memory operator M^inf_Θ(x_{1:T}).

    Z2 Reference: §11 of 02_rigorous_architecture.md
    Formal Claims: Thm 12.1 (causality), Thm 12.2 (bounded complexity),
                   Thm 12.3 (stability)

    Deploy mode (B6, §11): hard projection → anchor build → normalize → lift → topology
    Train mode (B7, §13): relaxed readout (y* used directly as soft weights)

    Parameters
    ----------
    trajectory : np.ndarray, shape (T, d)
        Input trajectory x_{1:T}.
    K : int
        Maximum anchor budget.
    r : int
        Refractory separation.
    lam : float
        Strong-concavity parameter (λ > 0).
    W_Theta : np.ndarray, shape (k, d+3)
        Learned lift matrix.
    Q : int
        Maximum homology degree.
    alpha : float, default 0.0
        Hysteresis gate. If 0.0, uses sharp event score.
    phi : callable, optional
        Feature map for hysteretic encoder.
    eta : callable, optional
        Scoring function for hysteretic encoder.
    max_edge_length : float, optional
        Maximum edge length for Rips complex.
    mu : np.ndarray, shape (d+3,), optional
        Pre-computed center for normalization. If None, computed from data.
    sigma : np.ndarray, shape (d+3,), optional
        Pre-computed scale for normalization. If None, computed from data.
    solver : str
        QP solver: "osqp" or "scipy".
    mode : str
        "deploy" is the supported deployment-time memory object.
        Train-time relaxed readout is implemented separately via
        ``synapse_core.training_readout.compute_relaxed_readout``.
    saliency_mode : str
        Causal saliency normalization mode from §4.
    saliency_temperature : float
        Temperature parameter for ``saliency_mode="temperature"``.

    Returns
    -------
    state : Z2MemoryState
        The complete Z2 memory state.
    """
    if trajectory.ndim != 2:
        raise ValueError(f"trajectory must have shape (T, d), got ndim={trajectory.ndim}")
    if trajectory.shape[0] < 1:
        raise ValueError("trajectory must have at least one timestep")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("trajectory must contain only finite values")
    if K < 0:
        raise ValueError(f"K must be non-negative, got {K}")
    if r < 0:
        raise ValueError(f"r must be non-negative, got {r}")
    if lam <= 0:
        raise ValueError(f"lam must be > 0, got {lam}")
    if mode not in {"deploy", "train"}:
        raise ValueError(f"mode must be 'deploy' or 'train', got {mode!r}")

    T, d = trajectory.shape
    D = d + 3
    if W_Theta.ndim != 2:
        raise ValueError(f"W_Theta must have shape (k, d+3), got ndim={W_Theta.ndim}")
    if not np.all(np.isfinite(W_Theta)):
        raise ValueError("W_Theta must contain only finite values")
    if W_Theta.shape[1] != D:
        raise ValueError(
            f"W_Theta second dimension must equal d+3={D}, got {W_Theta.shape[1]}"
        )
    k = W_Theta.shape[0]

    # Step 1: Event scoring (causal)
    if alpha == 0.0 and phi is None and eta is None:
        scores = sharp_event_score(trajectory)
    else:
        scores, _ = hysteretic_event_score(trajectory, alpha=alpha, phi=phi, eta=eta)

    saliency_scores = normalize_saliency(
        scores,
        mode=saliency_mode,
        temperature=saliency_temperature,
    )

    # Step 2: Relaxed selector QP (§5)
    y_star = solve_relaxed_selector(saliency_scores, K, r, lam, solver=solver)

    # Step 3: Hard projection (§6) — deploy mode
    #         In train mode (B7), we skip hard projection and use y* directly
    if mode != "deploy":
        raise NotImplementedError(
            "Use synapse_core.training_readout.compute_relaxed_readout for the §13 training-time object."
        )
    I_star = hard_projection(y_star, K, r)

    # Step 4: Anchor construction (§7)
    anchors = build_anchors(I_star, trajectory, scores)

    # Step 5: Anchor vector construction (§7–8)
    V = anchor_vectors(anchors, D=D)

    # Step 6: Normalization N(v) = D⁻¹(v − μ) (§8)
    V_norm, mu_used, sigma_used = normalize_anchors(V, mu=mu, sigma=sigma)

    # Step 7: Learned lift ρ_Θ(a_j) = W_Θ N(v(a_j)) (§9)
    P_Theta = apply_lift(V_norm, W_Theta)

    # Step 8: Persistent topological summary (§10)
    diagrams = compute_persistence_diagrams(P_Theta, Q, max_edge_length=max_edge_length)

    return Z2MemoryState(
        anchors=anchors,
        anchor_indices=I_star,
        y_star=y_star,
        V=V,
        V_norm=V_norm,
        point_cloud=P_Theta,
        persistence_diagrams=diagrams,
        event_scores=scores,
        mu=mu_used,
        sigma=sigma_used,
    )
