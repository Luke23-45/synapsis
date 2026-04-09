"""
Memory Operator — Formal Math Reference: §8 of 01_main_definition.md, §7 of 02_rigorous_architecture.md

The SYNAPSE memory operator:

    M(x_{1:T}) = ( A(x_{1:T}), Dgm_0(P(x_{1:T})), ..., Dgm_Q(P(x_{1:T})) )

This is the formal memory system. It composes:
    1. Event scoring (causal)
    2. Anchor selection (constrained optimization with LexMin)
    3. Weighted geometric lift
    4. Persistent topological summary

The codomain is:
    Y = ⋃_{m=0}^{K} (ℝ^{d+3})^m × ∏_{q=0}^{Q} D_q
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass

from .event_encoder import sharp_event_score, hysteretic_event_score
from .anchor_selector import select_anchors, Anchor
from .geometric_lift import lift_anchors
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
