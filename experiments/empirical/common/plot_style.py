"""
experiments/empirical/common/plot_style.py

Publication-quality matplotlib style configuration.

Phase 2 Shared Infrastructure — see docs/implementation/phase2_empirical_validation/08_libraries_and_dependencies.md §3
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt


def set_publication_style() -> None:
    """Apply a clean, publication-ready matplotlib style."""
    matplotlib.rcParams.update({
        "figure.figsize": (6, 4),
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
    })
