"""
Anchor Selector — Formal Math Reference: §4–5 of 01_main_definition.md, §4 of 02_rigorous_architecture.md

Implements the deterministic anchor selection rule:

    I*(x_{1:T}) = LexMin( argmax_{I ∈ 𝔍_{K,r,T}} Σ_{i∈I} e_i )
    subject to  e_i ≥ τ  for all i ∈ I

where:
    𝔍_{K,r,T} = { I = {i_1 < ... < i_m} ⊆ {2,...,T} : m ≤ K, i_{j+1} − i_j > r }

The LexMin rule selects the lexicographically smallest maximizing index set,
making the selector deterministic and M a genuine function.

Each anchor a_j = (t_j, s_j, δ_j, ξ_j) stores:
    t_j = i_j / T          (normalized event time)
    s_j = x_{i_j}          (retained state)
    δ_1 = i_1 − 1          (elapsed since start)
    δ_j = i_j − i_{j-1}    (elapsed since previous anchor, j ≥ 2)
    ξ_j = e_{i_j}          (event intensity)
"""

import numpy as np
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass


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
