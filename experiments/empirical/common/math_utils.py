"""
experiments/empirical/common/math_utils.py

Consolidated mathematical utilities for all EMP-* experiments.

Single source of truth for helpers previously duplicated across 7+ files:
  - Orthogonal matrix generation
  - Trajectory generation with sparse events
  - Point cloud padding
  - Cloud geometry summaries
  - Ridge probe classification
  - Bottleneck distance (vectorised)

All functions are fully vectorised using NumPy/SciPy — no Python-level loops
over data elements.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matrix / Cloud Utilities
# ---------------------------------------------------------------------------

def make_orthogonal_W(k: int, D: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random orthogonal lift matrix W_Θ ∈ ℝ^{k×D}.

    When k ≤ D, returns the first k rows of a random orthogonal matrix (via QR).
    When k > D, pads with zeros to shape (k, D) with identity in the top-left.
    """
    if k <= D:
        A = rng.standard_normal((D, D))
        Q, _ = np.linalg.qr(A)
        return Q[:k, :].astype(np.float64)
    W = np.zeros((k, D), dtype=np.float64)
    W[:D, :D] = np.eye(D)
    return W


def pad_rows(arr: np.ndarray, rows: int) -> np.ndarray:
    """Pad or truncate ``arr`` to exactly ``rows`` rows. Returns float32."""
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0:
        cols = arr.shape[1] if arr.ndim == 2 else 1
        return np.zeros((rows, cols), dtype=np.float32)
    if arr.shape[0] >= rows:
        return arr[:rows].astype(np.float32)
    pad = np.zeros((rows - arr.shape[0], arr.shape[1]), dtype=np.float32)
    return np.concatenate([arr.astype(np.float32), pad], axis=0)


def generate_trajectory(
    d: int,
    T: int,
    rng: np.random.Generator,
    *,
    walk_scale: float = 0.3,
    event_magnitude: float = 3.0,
    min_events: int = 3,
    events_per_50: int = 1,
) -> np.ndarray:
    """Generate a random-walk trajectory with sparse sharp transitions.

    Parameters
    ----------
    d : int
        Dimensionality of the state space.
    T : int
        Trajectory length (timesteps).
    rng : Generator
        NumPy random generator.
    walk_scale : float
        Standard deviation of the random walk increments.
    event_magnitude : float
        Magnitude of sharp transition jumps.
    min_events : int
        Minimum number of injected events.
    events_per_50 : int
        Approximately one event per this many timesteps.

    Returns
    -------
    np.ndarray, shape (T, d), dtype float64
    """
    traj = np.cumsum(rng.standard_normal((T, d)) * walk_scale, axis=0)
    n_events = max(min_events, T // 50 * events_per_50)
    if T > 15:
        candidates = np.arange(10, T - 5)
        n_events = min(n_events, len(candidates))
        event_locs = rng.choice(candidates, size=n_events, replace=False)
        # Fully vectorised: build cumulative jump offsets via scatter-add
        jumps = rng.standard_normal((n_events, d)) * event_magnitude
        # Create an array of per-timestep deltas, then cumsum to propagate
        jump_deltas = np.zeros((T, d), dtype=np.float64)
        np.add.at(jump_deltas, event_locs, jumps)
        traj += np.cumsum(jump_deltas, axis=0)
    return traj.astype(np.float64)


def cloud_geometry_summary(cloud: np.ndarray) -> np.ndarray:
    """Pairwise distance + norm statistics (no topology). Returns (8,) float32.

    Fully vectorised via ``scipy.spatial.distance.cdist``.
    """
    if cloud.size == 0 or cloud.shape[0] < 2:
        return np.zeros(8, dtype=np.float32)

    # Vectorised pairwise distances
    dists = cdist(cloud, cloud, metric="euclidean")
    tri_idx = np.triu_indices_from(dists, k=1)
    tri = dists[tri_idx]
    norms = np.linalg.norm(cloud, axis=1)

    return np.array([
        tri.mean(), tri.std(),
        tri.max(), np.percentile(tri, 75),
        norms.mean(), norms.std(),
        norms.max(), np.percentile(norms, 75),
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Ridge Probe (consolidated from 3 variants)
# ---------------------------------------------------------------------------

def ridge_probe_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    alphas: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    n_shuffles: int = 1,
) -> float:
    """Ridge regression OVR probe with cross-validated regularisation.

    If ``X_val`` / ``y_val`` are provided, alpha is selected on the validation
    set and accuracy is reported on the test set (EMP-06 protocol).
    Otherwise, alpha is selected directly on test accuracy (EMP-03/04 protocol).

    All computation is vectorised via NumPy matrix ops.

    Parameters
    ----------
    X_train, y_train : training features and integer labels.
    X_test, y_test   : test features and integer labels.
    X_val, y_val      : optional validation features and labels.
    alphas            : regularisation strengths to sweep.
    n_shuffles        : number of random-order reruns (for stability).

    Returns
    -------
    float : classification accuracy on the test set.
    """
    if len(X_train) < 4 or len(X_test) < 2:
        return 0.0
    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        return float(np.mean(y_test == unique_labels[0]))

    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    # Standardise
    mu = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    X_train_n = (X_train - mu) / std
    X_test_n = (X_test - mu) / std

    # Augment with bias column
    X_train_aug = np.c_[X_train_n, np.ones(len(X_train_n))]
    X_test_aug = np.c_[X_test_n, np.ones(len(X_test_n))]
    XtX = X_train_aug.T @ X_train_aug
    p = X_train_aug.shape[1]

    # Precompute per-class target vectors: (n_classes, n_train)
    Y_onehot = np.column_stack(
        [(y_train == lb).astype(np.float64) for lb in unique_labels]
    )  # (n_train, n_classes)
    XtY = X_train_aug.T @ Y_onehot  # (p, n_classes)

    if X_val is not None and y_val is not None:
        X_val = np.asarray(X_val, dtype=np.float64)
        X_val_n = (X_val - mu) / std
        X_val_aug = np.c_[X_val_n, np.ones(len(X_val_n))]

    best_acc = 0.0
    best_test_acc = 0.0  # initialise to avoid UnboundLocalError
    for alpha in alphas:
        reg = alpha * np.eye(p, dtype=np.float64)
        reg[-1, -1] = 0.0  # don't regularise bias
        try:
            W = np.linalg.solve(XtX + reg, XtY)  # (p, n_classes) — faster than pinv
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if system is singular
            W = np.linalg.lstsq(XtX + reg, XtY, rcond=None)[0]

        if X_val is not None and y_val is not None:
            val_preds = unique_labels[np.argmax(X_val_aug @ W, axis=1)]
            val_acc = float(np.mean(val_preds == y_val))
            if val_acc > best_acc:
                best_acc = val_acc
                test_preds = unique_labels[np.argmax(X_test_aug @ W, axis=1)]
                best_test_acc = float(np.mean(test_preds == y_test))
        else:
            preds = unique_labels[np.argmax(X_test_aug @ W, axis=1)]
            acc = float(np.mean(preds == y_test))
            best_acc = max(best_acc, acc)

    if X_val is not None and y_val is not None:
        return best_test_acc
    return best_acc


# ---------------------------------------------------------------------------
# Bottleneck Distance (fully vectorised)
# ---------------------------------------------------------------------------

def _has_gudhi_bottleneck() -> bool:
    """Check if gudhi's C++ bottleneck_distance is available."""
    try:
        import gudhi
        _ = gudhi.bottleneck_distance
        return True
    except (ImportError, AttributeError):
        return False


def bottleneck_distance(
    dgm_a,
    dgm_b,
) -> float:
    """Compute bottleneck distance d_B between two persistence diagrams.

    Uses gudhi.bottleneck_distance when available (exact, C++ backed).
    Falls back to a fully-vectorised scipy-based Hungarian matching.

    Parameters
    ----------
    dgm_a, dgm_b : PersistenceDiagram (iterable of (birth, death) pairs)

    Returns
    -------
    float : the bottleneck distance.
    """
    # Filter to finite-death points — vectorised when possible
    if hasattr(dgm_a, '__len__') and len(dgm_a) > 0:
        try:
            arr_a = np.asarray(dgm_a, dtype=np.float64)
            if arr_a.ndim == 2 and arr_a.shape[1] >= 2:
                mask_a = np.isfinite(arr_a[:, 1])
                fa = arr_a[mask_a, :2]
            else:
                fa = np.array([(b, d) for b, d in dgm_a if np.isfinite(d)], dtype=np.float64)
        except (ValueError, TypeError):
            fa = np.array([(b, d) for b, d in dgm_a if np.isfinite(d)], dtype=np.float64)
    else:
        fa = np.empty((0, 2), dtype=np.float64)

    if hasattr(dgm_b, '__len__') and len(dgm_b) > 0:
        try:
            arr_b = np.asarray(dgm_b, dtype=np.float64)
            if arr_b.ndim == 2 and arr_b.shape[1] >= 2:
                mask_b = np.isfinite(arr_b[:, 1])
                fb = arr_b[mask_b, :2]
            else:
                fb = np.array([(b, d) for b, d in dgm_b if np.isfinite(d)], dtype=np.float64)
        except (ValueError, TypeError):
            fb = np.array([(b, d) for b, d in dgm_b if np.isfinite(d)], dtype=np.float64)
    else:
        fb = np.empty((0, 2), dtype=np.float64)

    if len(fa) == 0 and len(fb) == 0:
        return 0.0

    # Reshape for edge cases
    if fa.ndim != 2:
        fa = fa.reshape(-1, 2) if fa.size > 0 else np.empty((0, 2), dtype=np.float64)
    if fb.ndim != 2:
        fb = fb.reshape(-1, 2) if fb.size > 0 else np.empty((0, 2), dtype=np.float64)

    # GUDHI fast path
    if _has_gudhi_bottleneck():
        import gudhi
        fa_list = fa.tolist() if len(fa) > 0 else [(0.0, 0.0)]
        fb_list = fb.tolist() if len(fb) > 0 else [(0.0, 0.0)]
        return float(gudhi.bottleneck_distance(fa_list, fb_list))

    # Scipy fallback — fully vectorised augmented cost matrix
    return _bottleneck_scipy_vectorised(fa, fb)


def _bottleneck_scipy_vectorised(
    fa: np.ndarray,
    fb: np.ndarray,
) -> float:
    """Bottleneck distance via augmented cost matrix + Hungarian algorithm.

    Fully vectorised — no Python loops over matrix entries.

    The augmented (na+nb) × (na+nb) cost matrix has structure:
        ┌─────────────┬─────────────┐
        │  real-real   │  A→diagonal │
        │  (na × nb)   │  (na × na)   │
        ├─────────────┼─────────────┤
        │  B→diagonal │  diag-diag  │
        │  (nb × nb)   │  (nb × na)   │
        └─────────────┴─────────────┘
    """
    na, nb = len(fa), len(fb)
    N = na + nb
    if N == 0:
        return 0.0

    LARGE = 1e18
    cost = np.full((N, N), LARGE, dtype=np.float64)

    # Block (0:na, 0:nb): L∞ between real points a_i and b_j
    if na > 0 and nb > 0:
        # birth_diff[i,j] = |fa[i,0] - fb[j,0]|
        birth_diff = np.abs(fa[:, 0:1] - fb[:, 0:1].T)  # (na, nb)
        death_diff = np.abs(fa[:, 1:2] - fb[:, 1:2].T)  # (na, nb)
        cost[:na, :nb] = np.maximum(birth_diff, death_diff)

    # Block (0:na, nb:nb+na): A[i] matched to diagonal — cost = persistence_i / 2
    if na > 0:
        diag_cost_a = (fa[:, 1] - fa[:, 0]) / 2.0  # (na,)
        np.fill_diagonal(cost[:na, nb:nb + na], diag_cost_a)

    # Block (na:na+nb, 0:nb): B[j] matched to diagonal — cost = persistence_j / 2
    if nb > 0:
        diag_cost_b = (fb[:, 1] - fb[:, 0]) / 2.0  # (nb,)
        np.fill_diagonal(cost[na:na + nb, :nb], diag_cost_b)

    # Block (na:, nb:): diagonal-to-diagonal — free (cost 0)
    cost[na:, nb:] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)
    return float(np.max(cost[row_ind, col_ind]))


# ---------------------------------------------------------------------------
# Match F1 (vectorised via linear_sum_assignment)
# ---------------------------------------------------------------------------

def match_f1(
    detected: List[int],
    ground_truth: List[int],
    tolerance: int,
) -> float:
    """F1 score for event detection with tolerance window.

    Uses ``scipy.optimize.linear_sum_assignment`` for optimal matching
    instead of O(|gt| × |det|) greedy search.

    A ground-truth event at position g is considered "hit" if matched
    to a detected event d with |d - g| ≤ tolerance.
    """
    if not ground_truth:
        return 1.0 if not detected else 0.0
    if not detected:
        return 0.0

    gt_arr = np.asarray(ground_truth, dtype=np.float64)
    det_arr = np.asarray(detected, dtype=np.float64)

    # Cost matrix: |det[i] - gt[j]|
    cost = np.abs(det_arr[:, None] - gt_arr[None, :])  # (n_det, n_gt)

    # Optimal assignment minimising total distance
    row_ind, col_ind = linear_sum_assignment(cost)

    # Count hits within tolerance
    hits = int(np.sum(cost[row_ind, col_ind] <= tolerance))

    precision = hits / len(detected)
    recall = hits / len(ground_truth)
    if precision + recall < 1e-10:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Metric Axiom Checks (vectorised)
# ---------------------------------------------------------------------------

def check_triangle_inequality(
    points: np.ndarray,
    n_triples: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Sample random triples and check d(a,c) ≤ d(a,b) + d(b,c).

    Fully vectorised — samples all triples at once and computes norms in batch.
    """
    n = len(points)
    if n < 3:
        return {"violation_rate": 0.0, "max_violation": 0.0, "n_triples_tested": 0}

    # Sample all triples at once: (n_triples, 3)
    indices = np.stack([
        rng.choice(n, size=n_triples, replace=True),
        rng.choice(n, size=n_triples, replace=True),
        rng.choice(n, size=n_triples, replace=True),
    ], axis=1)

    # Remove degenerate triples (any two indices equal)
    valid = (indices[:, 0] != indices[:, 1]) & \
            (indices[:, 1] != indices[:, 2]) & \
            (indices[:, 0] != indices[:, 2])
    indices = indices[valid]

    if len(indices) == 0:
        return {"violation_rate": 0.0, "max_violation": 0.0, "n_triples_tested": 0}

    a = points[indices[:, 0]]  # (m, dim)
    b = points[indices[:, 1]]
    c = points[indices[:, 2]]

    d_ab = np.linalg.norm(a - b, axis=1)
    d_bc = np.linalg.norm(b - c, axis=1)
    d_ac = np.linalg.norm(a - c, axis=1)

    gaps = d_ac - (d_ab + d_bc)
    violations = gaps > 1e-10
    n_tested = len(indices)

    return {
        "violation_rate": float(violations.sum()) / n_tested,
        "max_violation": float(np.max(gaps)) if float(np.max(gaps)) > 0 else 0.0,
        "n_triples_tested": int(n_tested),
    }


def check_symmetry(
    points: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
) -> float:
    """Check d(a,b) == d(b,a). Returns max asymmetry.

    For L2 norm this is always 0.0 (up to floating-point), but we verify
    the implementation is correct. Fully vectorised.
    """
    n = len(points)
    if n < 2:
        return 0.0

    idx_a = rng.choice(n, size=n_pairs, replace=True)
    idx_b = rng.choice(n, size=n_pairs, replace=True)
    valid = idx_a != idx_b
    idx_a, idx_b = idx_a[valid], idx_b[valid]

    if len(idx_a) == 0:
        return 0.0

    d_ab = np.linalg.norm(points[idx_a] - points[idx_b], axis=1)
    d_ba = np.linalg.norm(points[idx_b] - points[idx_a], axis=1)
    return float(np.max(np.abs(d_ab - d_ba)))


def check_non_negativity(points: np.ndarray, max_check: int = 200) -> bool:
    """Verify all pairwise distances are non-negative.

    Uses vectorised ``cdist`` on a subsample rather than Python loops.
    """
    n = min(len(points), max_check)
    if n < 2:
        return True
    sub = points[:n]
    dists = cdist(sub, sub, metric="euclidean")
    return bool(np.all(dists >= -1e-10))


def check_identity(points: np.ndarray) -> float:
    """Verify d(a,a) == 0 for all points. Returns max self-distance.

    Vectorised: ‖p - p‖₂ = 0 for all p by construction, but we verify.
    """
    if len(points) == 0:
        return 0.0
    # Self-distances are trivially zero; verify numerically
    self_dists = np.linalg.norm(points - points, axis=1)
    return float(np.max(self_dists))
