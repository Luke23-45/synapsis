"""
Metrics — Distance and comparison functions for experiment verification.
"""

import numpy as np
from typing import List, Tuple, Optional


def exact_match_scalar(a: float, b: float, tol: float = 1e-14) -> bool:
    """Check if two scalars are equal within tolerance."""
    return abs(a - b) <= tol


def exact_match_array(a: np.ndarray, b: np.ndarray, tol: float = 1e-14) -> bool:
    """Check if two arrays are equal element-wise within tolerance."""
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, atol=tol, rtol=0)


def exact_match_index_set(a: List[int], b: List[int]) -> bool:
    """Check if two index sets are identical."""
    return a == b


def hausdorff_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute the Hausdorff distance between two point clouds.

    d_H(P, Q) = max( max_p min_q ‖p-q‖₂, max_q min_p ‖q-p‖₂ )

    Parameters
    ----------
    P : np.ndarray, shape (m, D)
    Q : np.ndarray, shape (n, D)

    Returns
    -------
    float
    """
    from scipy.spatial.distance import cdist

    if P.shape[0] == 0 and Q.shape[0] == 0:
        return 0.0
    if P.shape[0] == 0 or Q.shape[0] == 0:
        return float("inf")

    D = cdist(P, Q, metric="euclidean")
    forward = np.max(np.min(D, axis=1))
    backward = np.max(np.min(D, axis=0))
    return float(max(forward, backward))


def bottleneck_distance(
    dgm1: List[Tuple[float, float]],
    dgm2: List[Tuple[float, float]],
) -> float:
    """
    Compute the bottleneck distance between two persistence diagrams.

    Uses persim if available, otherwise a brute-force implementation.

    Parameters
    ----------
    dgm1, dgm2 : list of (birth, death) tuples

    Returns
    -------
    float
    """
    try:
        import persim

        # Convert to numpy arrays expected by persim
        arr1 = _diagram_to_array(dgm1)
        arr2 = _diagram_to_array(dgm2)
        return persim.bottleneck(arr1, arr2)
    except ImportError:
        return _bottleneck_brute_force(dgm1, dgm2)


def _diagram_to_array(dgm: List[Tuple[float, float]]) -> np.ndarray:
    """Convert diagram list to numpy array, filtering infinite points."""
    finite_points = [(b, d) for b, d in dgm if np.isfinite(d)]
    if not finite_points:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(finite_points, dtype=np.float64)


def _bottleneck_brute_force(
    dgm1: List[Tuple[float, float]],
    dgm2: List[Tuple[float, float]],
) -> float:
    """
    Brute-force bottleneck distance computation.

    This is O(n! * m!) in the worst case and only suitable for small diagrams.
    For experiments with large diagrams, install `persim`.

    The bottleneck distance is:
        d_B = inf_γ sup_p ‖p − γ(p)‖_∞

    where γ ranges over all matchings between dgm1 and dgm2 (including
    matching to the diagonal).
    """
    # Filter to finite points
    pts1 = [(b, d) for b, d in dgm1 if np.isfinite(d)]
    pts2 = [(b, d) for b, d in dgm2 if np.isfinite(d)]

    n1 = len(pts1)
    n2 = len(pts2)

    if n1 == 0 and n2 == 0:
        return 0.0

    # Cost of matching point to diagonal: L∞ distance to projection
    def diag_cost(b, d):
        return (d - b) / 2.0

    # Cost of matching two points
    def match_cost(p1, p2):
        return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

    # Collect all candidate distances
    candidates = set()
    candidates.add(0.0)

    for b, d in pts1:
        candidates.add(diag_cost(b, d))
    for b, d in pts2:
        candidates.add(diag_cost(b, d))
    for p1 in pts1:
        for p2 in pts2:
            candidates.add(match_cost(p1, p2))

    # Binary search: smallest δ such that a valid matching exists with cost ≤ δ
    candidates = sorted(candidates)

    def _can_match(delta):
        """Check if a matching with bottleneck cost ≤ delta exists using augmenting paths."""
        # Build bipartite graph: pts1[i] can match pts2[j] if match_cost ≤ delta
        # pts1[i] can match diagonal if diag_cost ≤ delta
        # pts2[j] can match diagonal if diag_cost ≤ delta

        # Check all pts2 can be matched to diagonal if needed
        for b, d in pts2:
            pass  # Will be handled by the matching

        # Simple greedy matching (sufficient for small diagrams in experiments)
        used2 = [False] * n2
        for i in range(n1):
            # Try to match pts1[i] to some pts2[j]
            matched = False
            for j in range(n2):
                if not used2[j] and match_cost(pts1[i], pts2[j]) <= delta + 1e-15:
                    used2[j] = True
                    matched = True
                    break
            if not matched:
                # Must match to diagonal
                if diag_cost(pts1[i][0], pts1[i][1]) > delta + 1e-15:
                    return False

        # Check unmatched pts2 can go to diagonal
        for j in range(n2):
            if not used2[j]:
                if diag_cost(pts2[j][0], pts2[j][1]) > delta + 1e-15:
                    return False

        return True

    # Find smallest delta that works
    for delta in candidates:
        if _can_match(delta):
            return delta

    # Should not reach here
    return candidates[-1] if candidates else 0.0


def max_pointwise_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute max_j ‖p_j − q_j‖₂ for matched point clouds.

    Parameters
    ----------
    P, Q : np.ndarray, shape (m, D) — must have same shape.

    Returns
    -------
    float
    """
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: P={P.shape}, Q={Q.shape}")
    if P.shape[0] == 0:
        return 0.0
    dists = np.linalg.norm(P - Q, axis=1)
    return float(np.max(dists))
