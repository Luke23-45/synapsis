"""
Trajectory Generators — Synthetic data factories for experiments.

Each generator produces trajectories with known ground-truth properties
so that experiment verification has exact expected answers.
"""

import numpy as np
from typing import List, Optional, Tuple


def piecewise_constant(
    d: int,
    T: int,
    change_points: List[int],
    values: Optional[List[np.ndarray]] = None,
    jump_magnitude: float = 5.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate a piecewise-constant trajectory with known change points.

    The trajectory has value v_k on the interval [c_{k-1}, c_k) where
    c_0 = 0 and c_{m+1} = T.

    Parameters
    ----------
    d : int
        Ambient dimension.
    T : int
        Trajectory length.
    change_points : list of int
        0-indexed positions where changes occur. Must be in (0, T).
    values : list of ndarray, optional
        Values for each segment. If None, random values with specified
        jump magnitude are generated.
    jump_magnitude : float
        Magnitude of jumps when auto-generating values.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    trajectory : np.ndarray, shape (T, d)
        The piecewise-constant trajectory.
    change_points : list of int
        The ground-truth change points (echoed back).
    """
    rng = np.random.default_rng(seed)

    # Validate change points
    for cp in change_points:
        if cp <= 0 or cp >= T:
            raise ValueError(f"Change point {cp} must be in (0, {T}).")
    if sorted(change_points) != change_points:
        raise ValueError("Change points must be sorted in increasing order.")

    # Generate segment values
    n_segments = len(change_points) + 1
    if values is not None:
        if len(values) != n_segments:
            raise ValueError(
                f"Expected {n_segments} values, got {len(values)}."
            )
        seg_values = [np.asarray(v, dtype=np.float64) for v in values]
    else:
        seg_values = []
        current = rng.standard_normal(d) * jump_magnitude
        seg_values.append(current.copy())
        for _ in range(len(change_points)):
            # Each jump has magnitude ~ jump_magnitude
            direction = rng.standard_normal(d)
            direction = direction / np.linalg.norm(direction)
            current = current + direction * jump_magnitude
            seg_values.append(current.copy())

    # Build trajectory
    trajectory = np.zeros((T, d), dtype=np.float64)
    boundaries = [0] + change_points + [T]
    for i in range(n_segments):
        start = boundaries[i]
        end = boundaries[i + 1]
        trajectory[start:end] = seg_values[i]

    return trajectory, change_points


def random_walk(
    d: int,
    T: int,
    step_std: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a Gaussian random walk trajectory.

    x_1 = 0,  x_t = x_{t-1} + ε_t,  ε_t ~ N(0, σ²I)

    Parameters
    ----------
    d : int
        Ambient dimension.
    T : int
        Trajectory length.
    step_std : float
        Standard deviation of each step.
    seed : int, optional
        Random seed.

    Returns
    -------
    trajectory : np.ndarray, shape (T, d)
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, step_std, size=(T, d))
    steps[0] = 0.0
    return np.cumsum(steps, axis=0)


def constant_trajectory(
    d: int,
    T: int,
    value: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate a constant trajectory (x_t = c for all t).

    Parameters
    ----------
    d : int
        Ambient dimension.
    T : int
        Trajectory length.
    value : ndarray, optional
        The constant value. Default: zero vector.

    Returns
    -------
    trajectory : np.ndarray, shape (T, d)
    """
    if value is None:
        value = np.zeros(d, dtype=np.float64)
    else:
        value = np.asarray(value, dtype=np.float64)
    return np.tile(value, (T, 1))


def oscillatory(
    d: int,
    T: int,
    frequencies: Optional[List[float]] = None,
    amplitudes: Optional[List[float]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a trajectory with sinusoidal components.

    Each dimension i has: x_t[i] = A_i * sin(2π f_i t / T + φ_i)

    Parameters
    ----------
    d : int
        Ambient dimension.
    T : int
        Trajectory length.
    frequencies : list of float, optional
        Frequencies per dimension. Default: random.
    amplitudes : list of float, optional
        Amplitudes per dimension. Default: random.
    seed : int, optional
        Random seed.

    Returns
    -------
    trajectory : np.ndarray, shape (T, d)
    """
    rng = np.random.default_rng(seed)

    if frequencies is None:
        frequencies = rng.uniform(0.5, 5.0, size=d)
    if amplitudes is None:
        amplitudes = rng.uniform(0.5, 3.0, size=d)

    t = np.arange(T, dtype=np.float64)
    phases = rng.uniform(0, 2 * np.pi, size=d)

    trajectory = np.zeros((T, d), dtype=np.float64)
    for i in range(d):
        trajectory[:, i] = amplitudes[i] * np.sin(
            2 * np.pi * frequencies[i] * t / T + phases[i]
        )

    return trajectory


def adversarial_dense_events(
    d: int,
    T: int,
    K: int,
    r: int,
    tau: float,
    event_magnitude: float = 10.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a trajectory designed to maximally pack events within constraints.

    Creates a trajectory where events occur at every position that satisfies
    the spacing constraint r, with magnitudes well above threshold τ.

    Parameters
    ----------
    d : int
        Ambient dimension.
    T : int
        Trajectory length.
    K : int
        Maximum anchor budget.
    r : int
        Minimum refractory separation.
    tau : float
        Event score threshold.
    event_magnitude : float
        Magnitude of events (>> tau).
    seed : int, optional
        Random seed.

    Returns
    -------
    trajectory : np.ndarray, shape (T, d)
    """
    rng = np.random.default_rng(seed)
    trajectory = np.zeros((T, d), dtype=np.float64)

    # We want sharp event scores e_t = ‖x_t − x_{t-1}‖₂ to be large
    # at positions spaced by exactly r+1 apart, starting from position 1.
    # Fill as many as possible up to 2K events to stress-test K budget.
    base_value = rng.standard_normal(d) * 0.01
    trajectory[0] = base_value

    event_count = 0
    last_event = -r - 1  # Allows first event at position 1

    for t in range(1, T):
        if (t - last_event) > r and event_count < 2 * K:
            # Place an event: large jump
            direction = rng.standard_normal(d)
            direction = direction / np.linalg.norm(direction)
            trajectory[t] = trajectory[t - 1] + direction * event_magnitude
            last_event = t
            event_count += 1
        else:
            # No event: same as previous (constant)
            trajectory[t] = trajectory[t - 1]

    return trajectory


def collision_pair(
    d: int,
    T: int,
    K: int,
    r: int,
    tau: float,
    jump_magnitude: float = 5.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two distinct trajectories x ≠ y with M(x) = M(y).

    Strategy: Both trajectories share the same change points and the same
    values at those change points, but differ in the "quiet" segments
    (where all event scores are below τ). Since the memory operator only
    retains anchors at change points, the memory states will be identical.

    Parameters
    ----------
    d : int
        Ambient dimension.
    T : int
        Trajectory length.
    K : int
        Maximum anchor budget.
    r : int
        Minimum refractory separation.
    tau : float
        Event score threshold.
    jump_magnitude : float
        Magnitude of jumps at change points.
    seed : int, optional
        Random seed.

    Returns
    -------
    x : np.ndarray, shape (T, d)
        First trajectory.
    y : np.ndarray, shape (T, d)
        Second trajectory. x ≠ y but M(x) = M(y).
    """
    rng = np.random.default_rng(seed)

    # Choose change points with sufficient spacing
    change_points = []
    pos = r + 2  # First change point (ensure room for pre-segment)
    n_changes = min(K, (T - r - 2) // (r + 1))
    for i in range(n_changes):
        if pos < T:
            change_points.append(pos)
            pos += r + 1
        else:
            break

    if not change_points:
        # Fallback: very simple case
        change_points = [T // 2] if T > 2 else []

    # Generate shared segment values at change points
    n_segments = len(change_points) + 1
    seg_values = []
    current = rng.standard_normal(d) * jump_magnitude
    seg_values.append(current.copy())
    for _ in change_points:
        direction = rng.standard_normal(d)
        direction = direction / (np.linalg.norm(direction) + 1e-15)
        current = current + direction * jump_magnitude
        seg_values.append(current.copy())

    # Build x: standard piecewise constant
    x = np.zeros((T, d), dtype=np.float64)
    boundaries = [0] + change_points + [T]
    for i in range(n_segments):
        x[boundaries[i] : boundaries[i + 1]] = seg_values[i]

    # Build y: SAME values at change points and segment values,
    # but add tiny perturbations WITHIN quiet segments (not at change points)
    y = x.copy()
    perturbation_mag = tau * 0.01  # Much smaller than threshold
    for i in range(n_segments):
        start = boundaries[i]
        end = boundaries[i + 1]
        # Perturb interior of this segment (not the first point or change points)
        # Only perturb at positions where e_t would still be < tau
        for t in range(start + 1, end):
            if t not in change_points:
                # Add a small perturbation that won't create event scores ≥ tau
                perturb = rng.standard_normal(d) * perturbation_mag
                y[t] = seg_values[i] + perturb
                # Must ensure y[t] stays close enough that e_t for the NEXT step
                # doesn't go above tau either
        # Reset the value just before each change point to be exactly seg_values[i]
        # so the change point jump is identical
        if end < T and (i + 1) < n_segments:
            y[end - 1] = seg_values[i]

    return x, y
