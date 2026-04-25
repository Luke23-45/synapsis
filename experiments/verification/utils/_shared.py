"""Shared helpers for the Z2 verification experiments."""

from __future__ import annotations

import importlib.util
from itertools import product
from typing import Any, Dict, Iterator, Sequence

import numpy as np


class _ParameterGrid:
    """Sized Cartesian-product iterator for parameter sweeps.

    Supports ``len()`` so that progress bars can display ETA without
    the caller having to pass ``total`` explicitly.
    """

    def __init__(self, axes: dict[str, Sequence[Any]]) -> None:
        self._names = tuple(axes)
        self._values = tuple(axes[n] for n in self._names)
        self._size = 1
        for v in self._values:
            self._size *= len(v)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for combo in product(*self._values):
            yield dict(zip(self._names, combo))


def iter_parameter_grid(**axes: Sequence[Any]) -> _ParameterGrid:
    """Yield dictionaries over the Cartesian product of named parameter axes.

    Returns a sized iterable (supports ``len()``) so that progress
    bars can automatically compute ETA.
    """
    return _ParameterGrid(axes)


def make_orthogonal_lift(k: int, D: int, rng: np.random.Generator) -> np.ndarray:
    """Create an orthogonal lift matrix with shape ``(k, D)``."""
    if k <= D:
        q, _ = np.linalg.qr(rng.standard_normal((D, D)))
        return np.asarray(q[:k, :], dtype=np.float64)

    lift = np.zeros((k, D), dtype=np.float64)
    lift[:D, :D] = np.eye(D, dtype=np.float64)
    return lift


def spaced_indices(T: int, m: int, r: int, rng: np.random.Generator) -> list[int]:
    """Construct ``m`` indices in ``[1, T-1]`` with strict spacing greater than ``r``."""
    if m < 1:
        return []

    min_span = 1 + (m - 1) * (r + 1)
    if min_span >= T:
        step = max(r + 1, (T - 1) // max(m, 1))
        return [min(1 + j * step, T - 1) for j in range(m)]

    indices: list[int] = []
    current = 1
    for j in range(m):
        max_choice = T - 1 - (m - j - 1) * (r + 1)
        chosen = current if max_choice <= current else int(rng.integers(current, max_choice + 1))
        indices.append(chosen)
        current = chosen + r + 1
    return indices


def assign_at(base: np.ndarray, indices: Sequence[int], values: Any) -> np.ndarray:
    """Return a copy of ``base`` with ``values`` assigned at ``indices``."""
    out = np.array(base, copy=True)
    if indices:
        out[np.asarray(indices, dtype=np.intp)] = values
    return out


def separated(indices: Sequence[int], r: int) -> bool:
    """Check that every pair of indices is separated by more than ``r``."""
    if len(indices) < 2:
        return True
    sorted_idx = np.sort(np.asarray(indices, dtype=np.int64))
    return bool(np.all(np.diff(sorted_idx) > r))


def theorem_dominance_holds(
    y_star: np.ndarray,
    indices: Sequence[int],
    K: int,
    atol: float = 1e-10,
) -> bool:
    """Check the dominance condition from Theorem 6.3."""
    if not indices:
        return False

    selected = np.asarray(indices, dtype=np.intp)
    mask = np.ones(y_star.size, dtype=bool)
    mask[selected] = False
    mask[0] = False

    selected_min = float(np.min(y_star[selected]))
    outside_max = float(np.max(y_star[mask])) if np.any(mask) else 0.0

    strict = selected_min > outside_max + atol
    auxiliary = (len(indices) == K) or (outside_max <= atol)
    return bool(strict and auxiliary)


def refractory_excess(y: np.ndarray, r: int) -> float:
    """Return the maximum refractory-constraint violation."""
    if r <= 0 or y.size < 2:
        return 0.0

    max_violation = 0.0
    for offset in range(1, min(r + 1, y.size)):
        max_violation = max(max_violation, float(np.max(y[:-offset] + y[offset:] - 1.0)))
    return max(0.0, max_violation)


def bounded_point_perturbation(shape: tuple[int, int], epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """Sample pointwise perturbations with row norms bounded by ``epsilon``."""
    normals = rng.standard_normal(shape)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    directions = normals / np.maximum(norms, 1e-15)
    magnitudes = epsilon * rng.uniform(0.0, 1.0, size=(shape[0], 1))
    return directions * magnitudes


def pairwise_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Compute the Euclidean pairwise distance matrix."""
    if points.size == 0:
        return np.zeros((points.shape[0], points.shape[0]), dtype=np.float64)

    diffs = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def clouds_equal(a: np.ndarray, b: np.ndarray, atol: float = 1e-8) -> bool:
    """Shape-aware ``allclose`` helper."""
    return bool(a.shape == b.shape and (a.size == 0 or np.allclose(a, b, atol=atol)))


def resolve_selector_solver(requested: str) -> str:
    """Use SciPy directly when ``osqp`` is unavailable."""
    if requested != "osqp":
        return requested
    return "osqp" if importlib.util.find_spec("osqp") is not None else "scipy"


def run_standalone(
    caller_file: str,
    experiment_id: str,
    experiment_name: str,
    run_experiment_fn,
    config_key: str,
    project_root: str | None = None,
) -> int:
    """Run an experiment as a standalone executable script.

    Combines GibbsQ-style direct execution with SYNAPSE's structured
    reporting and capsule artifact management.  Each experiment script
    calls this from its ``if __name__ == "__main__"`` block.

    Parameters
    ----------
    caller_file : str
        The ``__file__`` of the calling script, used to resolve the
        project root for import path setup.
    experiment_id : str
        Short identifier (e.g. ``"VZ2-08"`` or ``"EZ2-01"``).
    experiment_name : str
        Human-readable experiment name.
    run_experiment_fn : callable
        The ``run_experiment(cfg, verbose) -> ExperimentReport`` function.
    config_key : str
        Key used for capsule directory naming and config overrides
        (e.g. ``"vz2_08_anchor_geometry_readout"``).
    project_root : str or None
        Explicit project root path.  If None, defaults to 3 levels
        above the caller (verification scripts).  Empirical scripts
        at 4+ levels deep should pass this explicitly.

    Returns
    -------
    int
        Exit code: 0 if PASS, 1 otherwise.
    """
    import argparse
    import inspect
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description=f"{experiment_id}: {experiment_name}",
    )
    parser.add_argument(
        "--config", default="experiments/configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-case results",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory (default: from config)",
    )
    parser.add_argument(
        "--no-record", action="store_true",
        help="Disable data recording (only save pass/fail report)",
    )
    args = parser.parse_args()

    # Ensure project root is importable
    if project_root is None:
        root = Path(caller_file).resolve().parent.parent.parent
    else:
        root = Path(project_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from experiments.utils.config import load_config, validate
    from experiments.utils.model_io import (
        create_run_capsule,
        save_config_snapshot,
        save_metrics,
        save_run_pointer,
    )
    from experiments.utils.logging import setup_run_logging
    from experiments.common.report import print_report, save_report_json

    # Load and validate configuration
    cfg = load_config(args.config)
    validate(cfg)

    # Create run capsule for artifact management
    output_dir = args.output_dir or cfg.output_dir
    capsule = create_run_capsule(output_dir, config_key)
    save_config_snapshot(capsule, cfg)

    # Setup per-run logging
    logger = setup_run_logging(
        capsule, verbose=args.verbose, experiment_id=experiment_id,
    )
    logger.info("Running %s: %s", experiment_id, experiment_name)
    logger.info("Config: %s", args.config)
    logger.info("Capsule: %s", capsule.root)

    # Create data recorder if the experiment function accepts it
    recorder = None
    if not args.no_record:
        sig = inspect.signature(run_experiment_fn)
        if "recorder" in sig.parameters:
            from experiments.verification.utils.data_recorder import VerificationRecorder
            recorder = VerificationRecorder(capsule, experiment_id)
            logger.info("Data recording enabled → %s", capsule.artifacts / "data")

    # Execute experiment
    kwargs = {"cfg": cfg, "verbose": args.verbose}
    if recorder is not None:
        kwargs["recorder"] = recorder
    report = run_experiment_fn(**kwargs)

    # Save results
    print_report(report)
    save_report_json(report, str(capsule.artifacts / "report.json"))
    save_metrics(capsule, {
        "experiment_id": report.experiment_id,
        "status": report.status,
        "passed": report.passed_cases,
        "total": report.total_cases,
        "duration": round(report.duration_seconds, 3),
    })
    save_run_pointer(capsule, output_dir)

    logger.info(
        "%s complete: %s (%d/%d cases passed, %.2fs)",
        experiment_id, report.status,
        report.passed_cases, report.total_cases,
        report.duration_seconds,
    )
    return 0 if report.status == "PASS" else 1

