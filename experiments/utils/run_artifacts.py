"""Helpers for standardized per-run artifact locations.

Adapted from GibbsQ ``gibbsq.utils.run_artifacts``.

Every experiment run writes to a capsule with this layout::

    outputs/<experiment_type>/<run_id>/
        logs/         -- run.log text
        figures/      -- PNG, PDF, SVG charts
        metrics/      -- metrics.jsonl, results.csv
        artifacts/    -- report.json, .npz data
        metadata/     -- config.yaml snapshot
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "logs_dir",
    "figures_dir",
    "metrics_dir",
    "artifacts_dir",
    "metadata_dir",
    "metadata_path",
    "config_path",
    "metrics_path",
    "figure_path",
]


def logs_dir(run_dir: Path) -> Path:
    """Return ``<run_dir>/logs``."""
    return Path(run_dir) / "logs"


def figures_dir(run_dir: Path) -> Path:
    """Return ``<run_dir>/figures``."""
    return Path(run_dir) / "figures"


def metrics_dir(run_dir: Path) -> Path:
    """Return ``<run_dir>/metrics``."""
    return Path(run_dir) / "metrics"


def artifacts_dir(run_dir: Path) -> Path:
    """Return ``<run_dir>/artifacts``."""
    return Path(run_dir) / "artifacts"


def metadata_dir(run_dir: Path) -> Path:
    """Return ``<run_dir>/metadata``."""
    return Path(run_dir) / "metadata"


def metadata_path(run_dir: Path, name: str) -> Path:
    """Return ``<run_dir>/metadata/<name>``."""
    return metadata_dir(run_dir) / name


def config_path(run_dir: Path) -> Path:
    """Return ``<run_dir>/metadata/config.yaml``."""
    return metadata_path(run_dir, "config.yaml")


def metrics_path(run_dir: Path, name: str = "metrics.jsonl") -> Path:
    """Return ``<run_dir>/metrics/<name>``."""
    return metrics_dir(run_dir) / name


def figure_path(run_dir: Path, stem: str) -> Path:
    """Return ``<run_dir>/figures/<stem>`` (no extension -- caller adds)."""
    return figures_dir(run_dir) / stem
