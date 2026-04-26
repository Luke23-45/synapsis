"""
Anchor Selector — Z2 Reference: §5–6 of 02_rigorous_architecture.md

Z2 Selector Pipeline:
    1. Relaxed selector: y* = argmax_{y ∈ Π_{K,r,T}} (sᵀy − λ‖y‖₂²)   [§5]
    2. Hard projection:  I* = Proj_{K,r,T}(y*)                       [§6]
    3. Anchor build:     A* = (a_1, ..., a_m) from I*                [§7]

Legacy Z1 selector (select_anchors) retained for backward compatibility.

Z2 Anchor a_j = (t_j, s_j, δ_j, ξ_j):
    t_j = i_j / T          (continuous normalized time, §7)
    s_j = x_{i_j}          (retained state)
    δ_1 = i_1 − 1          (discrete elapsed timesteps since start)
    δ_j = i_j − i_{j-1}    (discrete elapsed timesteps since prev anchor, j ≥ 2)
    ξ_j = e_{i_j}          (event intensity)
"""

import numpy as np
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass
import warnings


# QP solvers leave small positive artifacts at positions where the true
# optimum is exactly 0.  Artifact magnitude depends on the input: isolated
# zero-saliency positions yield ~1e-6, but when small saliency values exist
# elsewhere the solver can distribute ~1e-4 noise across many positions.
# Genuine activations satisfy y*_t ≥ s_min/(2λ), which is orders of magnitude
# above this threshold for any reasonable configuration.
# Used by solve_relaxed_selector and hard_projection.
_SOLVER_ZERO_TOL: float = 1e-4


@dataclass
class Anchor:
    """
    A single retained anchor.

    Formal definition (§5 of 01_main_definition.md):
        a_j = (t_j, s_j, δ_j, ξ_j)
    """

    t: float  # Normalized time t_j = i_j / T
    s: np.ndarray  # Retained state s_j = x_{i_j}
    delta: int  # Elapsed duration δ_j
    xi: float  # Event intensity ξ_j = e_{i_j}
    index: int  # Original index i_j (not part of formal tuple, but needed for verification)


def admissible_index_sets(K: int, r: int, T: int) -> List[List[int]]:
    """
    Enumerate all admissible index sets in 𝔍_{K,r,T}.

    Formal definition (§5 of 01_main_definition.md):
        𝔍_{K,r,T} = { I = {i_1 < ... < i_m} ⊆ {2,...,T}
                       : m ≤ K, i_{j+1} − i_j > r }

    Note: The empty set is always admissible (m=0 ≤ K).

    Parameters
    ----------
    K : int
        Maximum number of retained anchors.
    r : int
        Minimum separation between anchors (i_{j+1} − i_j > r).
    T : int
        Trajectory length.

    Returns
    -------
    sets : list of list of int
        All admissible index sets, each in sorted order.
    """
    if T < 2:
        return [[]]  # Only the empty set is admissible

    candidates = list(range(1, T))  # 0-indexed: positions 1..T-1 correspond to formal indices 2..T

    result = [[]]  # Empty set is always admissible

    def _recurse(start: int, current: List[int]):
        if len(current) == K:
            return
        for i in range(start, len(candidates)):
            idx = candidates[i]
            # Check spacing constraint with the last selected
            if current and (idx - current[-1]) <= r:
                continue
            new_set = current + [idx]
            result.append(list(new_set))
            _recurse(i + 1, new_set)

    _recurse(0, [])
    return result


def select_anchors(
    scores: np.ndarray,
    trajectory: np.ndarray,
    K: int,
    r: int,
    tau: float,
) -> Tuple[List[int], List[Anchor]]:
    """
    Select the optimal anchor index set I* and build the anchor sequence A.

    Formal definition (§5 of 01_main_definition.md):
        I*(x_{1:T}) = LexMin( argmax_{I ∈ 𝔍_{K,r,T}} Σ_{i∈I} e_i )
        subject to  e_i ≥ τ  for all i ∈ I

    This implementation uses dynamic programming for large T (efficient)
    and falls back to enumeration for small T (exact verification).

    Parameters
    ----------
    scores : np.ndarray, shape (T,)
        Event scores e_1, ..., e_T. Must satisfy e_1 = 0.
    trajectory : np.ndarray, shape (T, d)
        The input trajectory x_{1:T}.
    K : int
        Maximum number of retained anchors.
    r : int
        Minimum refractory separation (i_{j+1} − i_j > r).
    tau : float
        Minimum event score threshold (e_i ≥ τ for all i ∈ I).

    Returns
    -------
    I_star : list of int
        The optimal index set I* (0-indexed).
    anchors : list of Anchor
        The anchor sequence A(x_{1:T}).
    """
    T = scores.shape[0]

    if K <= 0 or T < 2:
        return [], []

    # Step 1: Identify feasible indices (0-indexed positions 1..T-1 where score ≥ τ)
    feasible = [i for i in range(1, T) if scores[i] >= tau]

    if not feasible:
        return [], []

    # Step 2: Find the maximum total score via DP, then extract the LexMin set
    I_star = _dp_lexmin_select(feasible, scores, K, r)

    # Step 3: Build anchor sequence
    anchors = _build_anchors(I_star, trajectory, scores, T)

    return I_star, anchors


def _dp_lexmin_select(
    feasible: List[int],
    scores: np.ndarray,
    K: int,
    r: int,
) -> List[int]:
    """
    Dynamic programming to find the lexicographically smallest index set
    among all that maximize the total score sum.

    The DP computes the maximum achievable score for each (position, count)
    state, then traces back greedily choosing the smallest index at each step.

    This guarantees LexMin among all maximizers, as required by the formal definition.
    """
    n = len(feasible)
    if n == 0:
        return []

    # dp[i][j] = maximum total score achievable by selecting exactly j indices
    #            from feasible[i:], with the constraint that the first selected
    #            index is feasible[i] or later.
    # We use -infinity to indicate impossible states.
    NEG_INF = -np.inf

    # dp[i][j]: max score from choosing j anchors from feasible[i:]
    # with spacing constraint relative to the "previous selected index"
    # We handle spacing by only considering transitions where gap > r.

    # Actually, let's use a cleaner formulation:
    # dp[i][j] = max total score from choosing exactly j indices from feasible[i:]
    #            where feasible[i] is the first candidate (no prior selection constraint)
    # To handle the gap constraint, when transitioning from feasible[i] to feasible[k],
    # we require feasible[k] − feasible[i] > r.

    # For LexMin: we want to choose the smallest feasible index at each step
    # among all choices that still allow achieving the maximum total score.

    # Step 1: Compute max achievable score for each (starting_position, anchors_remaining)
    # dp[i][j] = max score using j anchors from feasible[i..n-1] where feasible[i]
    #            IS selected (so spacing constraint applies from feasible[i] forward)

    # Precompute: for each feasible[i], the smallest index k > i such that
    # feasible[k] − feasible[i] > r
    next_valid = [0] * n
    for i in range(n):
        k = i + 1
        while k < n and feasible[k] - feasible[i] <= r:
            k += 1
        next_valid[i] = k

    # dp[i][j] = max score choosing j anchors starting from feasible[i] (inclusive)
    #            where feasible[i] is selected
    # dp[i][1] = scores[feasible[i]]
    # dp[i][j] = scores[feasible[i]] + max over k ∈ [next_valid[i]..n-1] of dp[k][j-1]

    # We want the global maximum: max over j=1..K, over i=0..n-1 of dp[i][j]
    # subject to j ≤ K

    # Allocate
    dp = np.full((n + 1, K + 1), NEG_INF, dtype=np.float64)
    # dp[n][0] = 0: choosing 0 anchors from an empty range
    dp[n, 0] = 0.0

    # Fill backwards
    # For i = n-1 down to 0:
    #   dp[i][0] = 0 (choosing nothing is always possible)
    #   dp[i][j] for j ≥ 1: either skip feasible[i] or select it
    #     - Skip: dp[i][j] = dp[i+1][j]  (NOTE: this is selecting j from feasible[i+1:] without selecting feasible[i])
    #     - Select: dp[i][j] = scores[feasible[i]] + dp[next_valid[i]][j-1]

    # Actually, we need a formulation where dp[i][j] = max score using j anchors from feasible[i:]
    # without requiring feasible[i] to be selected.

    for i in range(n, -1, -1):
        dp[i, 0] = 0.0

    for i in range(n - 1, -1, -1):
        for j in range(1, K + 1):
            # Option A: skip feasible[i]
            skip_val = dp[i + 1, j]

            # Option B: select feasible[i]
            nv = next_valid[i]
            select_val = scores[feasible[i]] + dp[nv, j - 1]

            dp[i, j] = max(skip_val, select_val)

    # Step 2: Find the maximum total score across all valid counts
    max_score = NEG_INF
    best_count = 0
    for j in range(K + 1):
        if dp[0, j] > max_score:
            max_score = dp[0, j]
            best_count = j

    if best_count == 0:
        return []

    # Step 3: Trace back to find the LexMin set
    # At each step, try the smallest feasible index first (greedy LexMin)
    result = []
    remaining = best_count
    pos = 0  # Current position in feasible array

    while remaining > 0 and pos < n:
        # Can we achieve the required remaining score by selecting feasible[pos]?
        nv = next_valid[pos]
        if remaining >= 1:
            select_score = scores[feasible[pos]] + (dp[nv, remaining - 1] if nv <= n else (0.0 if remaining - 1 == 0 else NEG_INF))
        else:
            select_score = NEG_INF

        # What's the target score we need to achieve from pos onwards?
        target = dp[pos, remaining]

        if abs(select_score - target) < 1e-12:
            # Selecting feasible[pos] still achieves the maximum — do it (LexMin: prefer earlier)
            result.append(feasible[pos])
            remaining -= 1
            pos = nv
        else:
            # Skip feasible[pos]
            pos += 1

    return result


def _build_anchors(
    I_star: List[int],
    trajectory: np.ndarray,
    scores: np.ndarray,
    T: int,
) -> List[Anchor]:
    """
    Build the anchor sequence A(x_{1:T}) from the selected indices.

    Formal definition (§5 of 01_main_definition.md):
        a_j = (t_j, s_j, δ_j, ξ_j) where
            t_j = i_j / T
            s_j = x_{i_j}
            δ_1 = i_1 − 1  (note: 0-indexed i_1 maps to formal index i_1+1, so δ_1 = i_1)
            δ_j = i_j − i_{j−1}  for j ≥ 2
            ξ_j = e_{i_j}

    Note on indexing: Our arrays are 0-indexed. The formal definition uses 1-indexed
    trajectories where indices run from 1 to T. In our implementation:
        - 0-indexed position i corresponds to formal index i+1
        - Formal t_j = i_j/T becomes (i+1)/T for 0-indexed i
        - Formal δ_1 = i_1 − 1 becomes i (the 0-indexed position) since formal i_1 = i+1
    """
    anchors = []
    for j, idx in enumerate(I_star):
        # t_j = (idx + 1) / T  [converting 0-indexed to formal 1-indexed]
        t_j = (idx + 1) / T

        # s_j = x_{i_j}
        s_j = trajectory[idx].copy()

        # δ_j
        if j == 0:
            # δ_1 = i_1 − 1 (formal), which is idx in 0-indexed
            delta_j = idx
        else:
            # δ_j = i_j − i_{j−1} (formal), same in 0-indexed: idx - I_star[j-1]
            delta_j = idx - I_star[j - 1]

        # ξ_j = e_{i_j}
        xi_j = scores[idx]

        anchors.append(Anchor(t=t_j, s=s_j, delta=delta_j, xi=xi_j, index=idx))

    return anchors


# =======================================================================
# Z2 Relaxed Selector — §5 of 02_rigorous_architecture.md
# =======================================================================

def solve_relaxed_selector(
    saliency: np.ndarray,
    K: int,
    r: int,
    lam: float,
    solver: str = "osqp",
) -> np.ndarray:
    """
    Solve the Z2 relaxed selector QP over polytope Π_{K,r,T}.

    Z2 Reference: §5 of 02_rigorous_architecture.md
    Formal Claims: Prop 5.1 (existence & uniqueness), Prop 5.2 (causality)

    The QP in standard form: minimize ½ yᵀPy + qᵀy
        P = 2λ I_T  (diagonal, PD for λ > 0)
        q = −s      (negated saliency)
    subject to:
        Σ y_t ≤ K                              (budget)
        y_t + y_u ≤ 1  for 1 ≤ |t−u| ≤ r     (refractory)
        0 ≤ y_t ≤ 1                            (box)
        y_1 = 0                                (first-step)

    Parameters
    ----------
    saliency : np.ndarray, shape (T,)
        Causal saliency scores s_{1:T}. Must satisfy s[0] = 0 (first step).
    K : int
        Maximum anchor budget (K ≥ 1).
    r : int
        Refractory separation (r ≥ 0).
    lam : float
        Strong-concavity parameter (λ > 0).
    solver : str
        QP solver backend: "osqp" or "scipy".

    Returns
    -------
    y_star : np.ndarray, shape (T,)
        The unique relaxed selector output y* ∈ Π_{K,r,T}.

    Raises
    ------
    ValueError
        If lam ≤ 0, K < 1, or saliency is empty.
    RuntimeError
        If the QP solver fails.
    """
    saliency = np.asarray(saliency, dtype=np.float64)
    if saliency.ndim != 1:
        raise ValueError(f"saliency must be one-dimensional, got shape {saliency.shape}")
    if not np.all(np.isfinite(saliency)):
        raise ValueError("saliency must contain only finite values")

    T = saliency.shape[0]

    if T < 1:
        raise ValueError("Saliency must have at least 1 timestep.")
    if K < 1:
        raise ValueError(f"K must be ≥ 1, got {K}")
    if lam <= 0:
        raise ValueError(f"λ must be > 0, got {lam}")

    # Trivial case: T = 1 → y = [0] (only feasible point)
    if T == 1:
        return np.array([0.0], dtype=np.float64)

    # Analytic fast-path for completely decoupled problem (M^inf with r=0)
    # When budget K >= T and refractory r = 0, the constraints decouple.
    if K >= T and r == 0:
        y_star = np.clip(saliency / (2.0 * lam), 0.0, 1.0)
        y_star[0] = 0.0
        y_star[y_star < _SOLVER_ZERO_TOL] = 0.0
        return y_star

    # y[0] is fixed to 0 by construction, so its saliency must not leak into
    # the linear term. Keeping a large first-step score here is mathematically
    # irrelevant but can destabilize downstream solvers numerically.
    saliency = saliency.copy()
    saliency[0] = 0.0

    # Build QP: minimize ½ yᵀPy + qᵀy
    P = 2.0 * lam * np.eye(T, dtype=np.float64)
    q = -saliency.astype(np.float64)

    # --- Inequality constraints: A_ub @ y ≤ b_ub ---
    rows_A = []
    rows_b = []

    # Budget: Σ y_t ≤ K
    rows_A.append(np.ones(T))
    rows_b.append(float(K))

    # Refractory: y_t + y_u ≤ 1 for 1 ≤ |t−u| ≤ r
    for t in range(T):
        for u in range(t + 1, min(t + r + 1, T)):
            row = np.zeros(T)
            row[t] = 1.0
            row[u] = 1.0
            rows_A.append(row)
            rows_b.append(1.0)

    A_ub = np.array(rows_A, dtype=np.float64) if rows_A else np.empty((0, T), dtype=np.float64)
    b_ub = np.array(rows_b, dtype=np.float64) if rows_b else np.empty(0, dtype=np.float64)

    # --- Equality constraint: y_1 = 0 (0-indexed: y[0] = 0) ---
    A_eq = np.zeros((1, T), dtype=np.float64)
    A_eq[0, 0] = 1.0
    b_eq = np.array([0.0], dtype=np.float64)

    # --- Bounds: 0 ≤ y_t ≤ 1 ---
    bounds = [(0.0, 1.0)] * T

    # Solve QP
    if solver == "osqp":
        y_star = _solve_osqp(P, q, A_ub, b_ub, A_eq, b_eq, bounds, T)
    else:
        y_star = _solve_scipy(P, q, A_ub, b_ub, A_eq, b_eq, bounds, T)

    # Numerical cleanup: QP solvers leave small positive artifacts at
    # positions where the true optimum is exactly 0 (lower bound active).
    # Snapping these to 0 restores the mathematical structure of the solution
    # and prevents spurious entries in the positive support used by hard_projection.
    y_star[y_star < _SOLVER_ZERO_TOL] = 0.0

    y_star[0] = 0.0
    return y_star


_osqp_cache = {}

def _solve_osqp(
    P: np.ndarray,
    q: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    bounds: list,
    T: int,
) -> np.ndarray:
    """Solve QP using OSQP."""
    try:
        import osqp
    except ImportError:
        warnings.warn("osqp not installed, falling back to scipy", RuntimeWarning)
        try:
            return _solve_scipy(P, q, A_ub, b_ub, A_eq, b_eq, bounds, T)
        except RuntimeError:
            warnings.warn("scipy fallback failed; falling back to cvxpy", RuntimeWarning)
            return _solve_cvxpy(P, q, A_ub, b_ub, A_eq, b_eq, bounds, T)

    global _osqp_cache
    # Cache key based on the problem dimensions (T) and constraints
    # K and r are captured implicitly via the shapes and contents of A_ub, b_ub
    # lam is captured via P[0,0]
    cache_key = (T, float(P[0, 0]), float(b_ub[0]) if b_ub.size else 0.0, A_ub.shape[0])

    if cache_key not in _osqp_cache:
        # OSQP form: minimize ½ xᵀPx + qᵀx  s.t. l ≤ Ax ≤ u
        n_ineq = A_ub.shape[0] if A_ub.size else 0
        n_eq = A_eq.shape[0] if A_eq.size else 0

        A_list = []
        l_list = []
        u_list = []

        if n_ineq > 0:
            A_list.append(A_ub)
            l_list.append(np.full(n_ineq, -np.inf))
            u_list.append(b_ub)

        if n_eq > 0:
            A_list.append(A_eq)
            l_list.append(b_eq)
            u_list.append(b_eq)

        A_list.append(np.eye(T))
        l_list.append(np.zeros(T))
        u_list.append(np.ones(T))

        A_osqp = np.vstack(A_list)
        l_osqp = np.concatenate(l_list)
        u_osqp = np.concatenate(u_list)

        from scipy.sparse import csc_matrix
        P_sparse = csc_matrix(P)
        A_sparse = csc_matrix(A_osqp)

        prob = osqp.OSQP()
        prob.setup(P_sparse, q, A_sparse, l_osqp, u_osqp,
                   verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=10000)
        _osqp_cache[cache_key] = prob
    else:
        prob = _osqp_cache[cache_key]
        prob.update(q=q)

    res = prob.solve()

    if res.info.status_val not in (1, 2):  # 1 = solved, 2 = solved inaccurate
        warnings.warn(
            f"OSQP solver failed with status: {res.info.status}. "
            f"Falling back to scipy.",
            RuntimeWarning
        )
        try:
            return _solve_scipy(P, q, A_ub, b_ub, A_eq, b_eq, bounds, T)
        except RuntimeError:
            warnings.warn("scipy fallback failed; falling back to cvxpy", RuntimeWarning)
            return _solve_cvxpy(P, q, A_ub, b_ub, A_eq, b_eq, bounds, T)

    return res.x.astype(np.float64)


_cvxpy_cache = {}

def _solve_cvxpy(
    P: np.ndarray,
    q: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    bounds: list,
    T: int,
) -> np.ndarray:
    """Solve QP using CVXPY (extremely fast fallback when OSQP is missing)."""
    try:
        import cvxpy as cp
    except ImportError:
        raise RuntimeError(
            "Neither 'osqp' nor 'cvxpy' is installed. The Z2 Relaxed Selector requires "
            "a mathematical solver for empirical verification. Please run: pip install osqp cvxpy"
        )

    global _cvxpy_cache
    cache_key = (T, float(P[0, 0]), float(b_ub[0]) if b_ub.size else 0.0, A_ub.shape[0])

    if cache_key not in _cvxpy_cache:
        y = cp.Variable(T)
        q_param = cp.Parameter(T)
        
        # P is diagonal (2*lam*I), so 0.5 * y^T P y == lam * sum(y^2)
        lam_val = P[0, 0] / 2.0
        objective = cp.Minimize(lam_val * cp.sum_squares(y) + q_param @ y)
        
        constraints = [y >= 0, y <= 1]
        
        if A_ub.size > 0:
            constraints.append(A_ub @ y <= b_ub)
        if A_eq.size > 0:
            constraints.append(A_eq @ y == b_eq)
            
        prob = cp.Problem(objective, constraints)
        _cvxpy_cache[cache_key] = (prob, y, q_param)
    else:
        prob, y, q_param = _cvxpy_cache[cache_key]

    q_param.value = q
    try:
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            prob.solve(warm_start=True)
            
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"CVXPY failed with status: {prob.status}")
            
        return np.clip(y.value, 0.0, 1.0).astype(np.float64)
    except Exception as e:
        raise RuntimeError(f"CVXPY fallback solver failed: {e}")


def _solve_scipy(
    P: np.ndarray,
    q: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    bounds: list,
    T: int,
) -> np.ndarray:
    """Solve the selector QP with SciPy as a robust fallback."""
    try:
        from scipy.optimize import Bounds, LinearConstraint, minimize
    except ImportError as e:
        raise RuntimeError(
            "SciPy is required for the relaxed-selector fallback but is not installed."
        ) from e

    def objective(y: np.ndarray) -> float:
        return 0.5 * float(y @ P @ y) + float(q @ y)

    def gradient(y: np.ndarray) -> np.ndarray:
        return (P @ y) + q

    constraints = []
    if A_ub.size > 0:
        constraints.append(
            LinearConstraint(A_ub, -np.inf * np.ones_like(b_ub), b_ub)
        )
    if A_eq.size > 0:
        constraints.append(LinearConstraint(A_eq, b_eq, b_eq))

    lower = np.array([b[0] for b in bounds], dtype=np.float64)
    upper = np.array([b[1] for b in bounds], dtype=np.float64)
    scipy_bounds = Bounds(lower, upper)

    y0 = np.clip(-q / np.diag(P), lower, upper)
    if A_eq.size > 0:
        y0[0] = b_eq[0]
    if A_ub.size > 0:
        budget = float(y0.sum())
        if budget > b_ub[0]:
            y0 *= float(b_ub[0] / max(budget, 1e-12))
            if A_eq.size > 0:
                y0[0] = b_eq[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        result = minimize(
            objective,
            y0,
            method="SLSQP",
            jac=gradient,
            bounds=scipy_bounds,
            constraints=constraints,
            options={
                "maxiter": 1000,
                "ftol": 1e-9,
                "disp": False,
            },
        )

    if not result.success:
        raise RuntimeError(
            f"SciPy solver failed: {result.message}"
        )

    y_star = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    if A_eq.size > 0:
        y_star[0] = b_eq[0]
    return y_star


# =======================================================================
# Z2 Hard Projection — §6 of 02_rigorous_architecture.md
# =======================================================================

def hard_projection(
    y_star: np.ndarray,
    K: int,
    r: int,
) -> List[int]:
    """
    Deterministic hard projection Proj_{K,r,T}(y*).

    Z2 Reference: §6 of 02_rigorous_architecture.md
    Formal Claims: Prop 6.1 (determinism & budget), Prop 6.2 (separation)

    Algorithm:
        1. Compute supp⁺(y*) = {t ∈ {2,...,T} : y*_t > 0}
        2. Sort indices by decreasing y*_t, ties broken by **smaller t**
           (Prop 6.1: determinism via smaller-t tiebreak)
        3. Greedily accept indices that satisfy |i−j| > r from all
           previously retained indices, until K selections made or
           positive-support exhausted.

    Parameters
    ----------
    y_star : np.ndarray, shape (T,)
        Relaxed selector output y* ∈ [0,1]^T.
    K : int
        Maximum anchor budget.
    r : int
        Refractory separation.

    Returns
    -------
    I_star : list of int
        Retained indices (0-indexed), sorted in increasing order.
        |I_star| ≤ K always (Prop 6.1).
    """
    T = len(y_star)
    if T < 2 or K < 1:
        return []

    # Step 1: supp⁺(y*) — indices t ∈ {1,...,T-1} (0-indexed) where y*_t > 0
    # Formal: t ∈ {2,...,T} (1-indexed) where y*_t > 0
    # 0-indexed: positions 1..T-1 where y_star[t] > 0
    positive_support = [t for t in range(1, T) if y_star[t] > _SOLVER_ZERO_TOL]

    if not positive_support:
        return []

    # Step 2: Sort by decreasing y*_t, ties broken by smaller t (Prop 6.1)
    sorted_indices = sorted(positive_support, key=lambda t: (-y_star[t], t))

    # Step 3: Greedy accept with separation |i−j| > r (Prop 6.2)
    retained = []
    for idx in sorted_indices:
        if len(retained) >= K:
            break
        # Check separation against all previously retained indices
        if all(abs(idx - ret) > r for ret in retained):
            retained.append(idx)

    # Return sorted in increasing order (formal I* = {i_1 < ... < i_m})
    retained.sort()
    return retained


# =======================================================================
# Z2 Anchor Construction — §7 of 02_rigorous_architecture.md
# =======================================================================

def build_anchors(
    I_star: List[int],
    trajectory: np.ndarray,
    event_scores: np.ndarray,
) -> List[Anchor]:
    """
    Build the Z2 anchor sequence A* from the retained index set I*.

    Z2 Reference: §7 of 02_rigorous_architecture.md

    For each i_j ∈ I* = {i_1 < ... < i_m}:
        t_j = i_j / T          (continuous normalized time)
        s_j = x_{i_j}          (retained state)
        δ_1 = i_1 − 1          (discrete elapsed timesteps)
        δ_j = i_j − i_{j-1}    (discrete elapsed timesteps, j ≥ 2)
        ξ_j = e_{i_j}          (event intensity)

    Parameters
    ----------
    I_star : list of int
        Retained indices (0-indexed), sorted in increasing order.
    trajectory : np.ndarray, shape (T, d)
        Input trajectory x_{1:T}.
    event_scores : np.ndarray, shape (T,)
        Event scores e_{1:T}.

    Returns
    -------
    anchors : list of Anchor
        The anchor sequence A*(x_{1:T}).
    """
    T = trajectory.shape[0]
    if not I_star:
        return []

    anchors = []
    for j, idx in enumerate(I_star):
        # t_j = (idx + 1) / T  [0-indexed idx → formal 1-indexed i_j = idx+1]
        t_j = (idx + 1) / T

        # s_j = x_{i_j}
        s_j = trajectory[idx].copy()

        # δ_j — DISCRETE elapsed timesteps (§7: "δ_j counts elapsed timesteps")
        if j == 0:
            # δ_1 = i_1 − 1 (formal), which is idx in 0-indexed
            delta_j = idx
        else:
            # δ_j = i_j − i_{j-1} (formal), same in 0-indexed
            delta_j = idx - I_star[j - 1]

        # ξ_j = e_{i_j}
        xi_j = event_scores[idx]

        anchors.append(Anchor(t=t_j, s=s_j, delta=delta_j, xi=xi_j, index=idx))

    return anchors
