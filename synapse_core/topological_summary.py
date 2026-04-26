"""
Topological Summary — Formal Math Reference: §7 of 01_main_definition.md, §6 of 02_rigorous_architecture.md

Implements the persistent topological summary of the lifted anchor cloud:

    For each ε ≥ 0, form the Vietoris-Rips complex VR_ε(P(x_{1:T})).
    For each homology degree q = 0, 1, ..., Q, compute:
        Dgm_q(P(x_{1:T}))

If P = ∅, all persistence diagrams are defined to be empty.

Dependencies:
    Uses `gudhi` for Vietoris-Rips complex and persistent homology.
    Falls back to a minimal internal implementation if gudhi is not available.
"""

import numpy as np
from typing import List, Optional, Tuple
import warnings


# Persistence diagram type: list of (birth, death) pairs
PersistenceDiagram = List[Tuple[float, float]]


def _check_gudhi_available() -> bool:
    """Check if gudhi is available for import."""
    try:
        import gudhi
        return True
    except ImportError:
        return False


def _check_ripser_available() -> bool:
    """Check if ripser is available for import."""
    try:
        import ripser
        return True
    except ImportError:
        return False


def has_full_persistence_backend() -> bool:
    """Return True when a higher-dimensional persistence backend is available."""
    return _check_gudhi_available() or _check_ripser_available()


def compute_persistence_diagrams(
    cloud: np.ndarray,
    Q: int,
    max_edge_length: Optional[float] = None,
) -> List[PersistenceDiagram]:
    """
    Compute persistence diagrams for the lifted anchor cloud.

    Formal definition (§7 of 01_main_definition.md):
        For each q = 0, 1, ..., Q:
            Dgm_q(P(x_{1:T}))
        computed from the Vietoris-Rips filtration of P.

    If P = ∅ (cloud has 0 points), returns Q+1 empty diagrams.

    Parameters
    ----------
    cloud : np.ndarray, shape (m, D) or (0, 0)
        The lifted anchor cloud P(x_{1:T}).
    Q : int
        Maximum homology degree retained (0, 1, ..., Q).
    max_edge_length : float, optional
        Maximum edge length for the Rips complex.
        If None, uses the diameter of the point cloud.

    Returns
    -------
    diagrams : list of PersistenceDiagram
        Q+1 persistence diagrams, one per homology degree 0..Q.
        Each diagram is a list of (birth, death) tuples.
    """
    if Q < 0:
        raise ValueError(f"Q must be non-negative, got {Q}")

    # Handle empty cloud: return Q+1 empty diagrams
    if cloud.size == 0 or cloud.shape[0] == 0:
        return [[] for _ in range(Q + 1)]

    m = cloud.shape[0]

    # Single point: H_0 has one infinite bar, all higher H_q are empty
    if m == 1:
        diagrams = [[] for _ in range(Q + 1)]
        diagrams[0] = [(0.0, float("inf"))]
        return diagrams

    # Compute maximum edge length if not provided
    if max_edge_length is None:
        from scipy.spatial.distance import pdist
        dists = pdist(cloud)
        max_edge_length = float(np.max(dists)) * 1.1 if len(dists) > 0 else 1.0

    # Try gudhi first, then ripser, then minimal fallback
    if _check_gudhi_available():
        return _compute_with_gudhi(cloud, Q, max_edge_length)
    elif _check_ripser_available():
        return _compute_with_ripser(cloud, Q, max_edge_length)
    else:
        return _compute_minimal_fallback(cloud, Q, max_edge_length)


def _compute_with_gudhi(
    cloud: np.ndarray,
    Q: int,
    max_edge_length: float,
) -> List[PersistenceDiagram]:
    """Compute persistence using GUDHI library."""
    import gudhi

    rips = gudhi.RipsComplex(points=cloud.tolist(), max_edge_length=max_edge_length)
    simplex_tree = rips.create_simplex_tree(max_dimension=Q + 1)
    simplex_tree.compute_persistence()

    diagrams = [[] for _ in range(Q + 1)]

    for dim, (birth, death) in simplex_tree.persistence():
        if dim <= Q:
            diagrams[dim].append((float(birth), float(death)))

    return diagrams


def _compute_with_ripser(
    cloud: np.ndarray,
    Q: int,
    max_edge_length: float,
) -> List[PersistenceDiagram]:
    """Compute persistence using ripser library."""
    from ripser import ripser

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='The input point cloud has more columns than rows; did you mean to transpose?',
            category=UserWarning,
        )
        warnings.filterwarnings(
            'ignore',
            message='The input matrix is square, but the distance_matrix flag is off.  Did you mean to indicate that this was a distance matrix?',
            category=UserWarning,
        )
        result = ripser(cloud, maxdim=Q, thresh=max_edge_length, distance_matrix=False)

    diagrams = [[] for _ in range(Q + 1)]
    for q in range(min(Q + 1, len(result["dgms"]))):
        for birth, death in result["dgms"][q]:
            diagrams[q].append((float(birth), float(death)))

    return diagrams


def _compute_minimal_fallback(
    cloud: np.ndarray,
    Q: int,
    max_edge_length: float,
) -> List[PersistenceDiagram]:
    """
    Minimal fallback: compute H_0 persistence from the distance matrix
    using single-linkage clustering (equivalent to VR H_0).

    For H_q with q ≥ 1, returns empty diagrams with a warning.
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage

    warnings.warn(
        "Neither gudhi nor ripser is installed. "
        "Computing only H_0 via single-linkage. H_q for q≥1 will be empty. "
        "Install gudhi or ripser for full persistent homology.",
        RuntimeWarning,
    )

    m = cloud.shape[0]
    dists = pdist(cloud)

    # Single-linkage clustering gives H_0 persistence
    Z = linkage(dists, method="single")

    diagrams = [[] for _ in range(Q + 1)]

    # H_0: Each point is born at 0. Components merge at the linkage distances.
    # There are m points, so m-1 merges. The last component lives to infinity.
    merge_dists = sorted(Z[:, 2])

    # m-1 finite bars (one dies at each merge distance)
    for dist in merge_dists:
        diagrams[0].append((0.0, float(dist)))

    # 1 infinite bar for the final connected component
    diagrams[0].append((0.0, float("inf")))

    return diagrams


def hausdorff_distance(P: np.ndarray, Q_cloud: np.ndarray) -> float:
    """
    Compute the Hausdorff distance between two point clouds.

    d_H(P, Q) = max( max_p min_q ‖p−q‖, max_q min_p ‖q−p‖ )

    Parameters
    ----------
    P, Q_cloud : np.ndarray, shape (m, D) and (n, D)

    Returns
    -------
    dist : float
    """
    from scipy.spatial.distance import cdist

    if P.shape[0] == 0 or Q_cloud.shape[0] == 0:
        return float("inf") if P.shape[0] != Q_cloud.shape[0] else 0.0

    D = cdist(P, Q_cloud, metric="euclidean")
    forward = np.max(np.min(D, axis=1))
    backward = np.max(np.min(D, axis=0))
    return float(max(forward, backward))
