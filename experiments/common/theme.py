"""
Publication-quality theme management for SYNAPSE experiment plots.

Adapted from GibbsQ's theme.py — provides:
- 'publication': White background for papers/reports
- 'dark': Dark background for presentations/screens

Uses Okabe-Ito colorblind-safe palette throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib import font_manager

__all__ = [
    "apply_theme",
    "get_current_theme",
    "THEMES",
    "ThemeConfig",
    "OKABE_ITO",
]

# Okabe-Ito colorblind-safe palette
OKABE_ITO = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#999999",  # Gray
]


@dataclass
class ThemeConfig:
    """Configuration for a matplotlib theme."""
    name: str
    figure_facecolor: str
    axes_facecolor: str
    text_color: str
    axes_edgecolor: str
    grid_color: str
    grid_alpha: float
    font_family: str
    font_serif: List[str]
    font_size_title: int
    font_size_label: int
    font_size_tick: int
    font_size_legend: int
    linewidth_axes: float
    linewidth_grid: float
    linewidth_lines: float
    dpi_figure: int
    dpi_save: int
    color_palette: List[str]


DARK_THEME = ThemeConfig(
    name="dark",
    figure_facecolor="#1a1a2e",
    axes_facecolor="#1a1a2e",
    text_color="white",
    axes_edgecolor="white",
    grid_color="white",
    grid_alpha=0.15,
    font_family="sans-serif",
    font_serif=["Inter", "Roboto", "Helvetica Neue", "Arial", "DejaVu Sans"],
    font_size_title=14,
    font_size_label=12,
    font_size_tick=10,
    font_size_legend=10,
    linewidth_axes=1.2,
    linewidth_grid=0.5,
    linewidth_lines=1.5,
    dpi_figure=150,
    dpi_save=300,
    color_palette=OKABE_ITO,
)

PUBLICATION_THEME = ThemeConfig(
    name="publication",
    figure_facecolor="white",
    axes_facecolor="white",
    text_color="black",
    axes_edgecolor="black",
    grid_color="gray",
    grid_alpha=0.3,
    font_family="serif",
    font_serif=["Times New Roman", "Computer Modern Roman", "DejaVu Serif", "serif"],
    font_size_title=12,
    font_size_label=11,
    font_size_tick=10,
    font_size_legend=9,
    linewidth_axes=0.8,
    linewidth_grid=0.5,
    linewidth_lines=1.2,
    dpi_figure=150,
    dpi_save=600,
    color_palette=OKABE_ITO,
)

THEMES = {
    "dark": DARK_THEME,
    "publication": PUBLICATION_THEME,
}

_current_theme: Optional[str] = None


def apply_theme(theme: str = "publication") -> None:
    """
    Apply a matplotlib theme for all subsequent plots.

    Parameters
    ----------
    theme : str
        Theme name ('dark' or 'publication').
    """
    global _current_theme

    if theme not in THEMES:
        raise ValueError(f"Unknown theme '{theme}'. Available: {list(THEMES.keys())}")

    _current_theme = theme
    config = THEMES[theme]

    installed_fonts = {f.name for f in font_manager.fontManager.ttflist}

    if config.font_family == "serif":
        available = [f for f in config.font_serif if f in installed_fonts]
        font_stack = available + ["DejaVu Serif", "serif"]
    else:
        available = [f for f in config.font_serif if f in installed_fonts]
        font_stack = available + ["DejaVu Sans", "sans-serif"]

    params = {
        "figure.facecolor": config.figure_facecolor,
        "figure.dpi": config.dpi_figure,
        "figure.figsize": (8, 5),

        "axes.facecolor": config.axes_facecolor,
        "axes.edgecolor": config.axes_edgecolor,
        "axes.linewidth": config.linewidth_axes,
        "axes.titlesize": config.font_size_title,
        "axes.labelsize": config.font_size_label,
        "axes.labelcolor": config.text_color,
        "axes.titlecolor": config.text_color,
        "axes.grid": True,

        "grid.color": config.grid_color,
        "grid.alpha": config.grid_alpha,
        "grid.linestyle": "--",
        "grid.linewidth": config.linewidth_grid,

        "xtick.labelsize": config.font_size_tick,
        "ytick.labelsize": config.font_size_tick,
        "xtick.color": config.text_color,
        "ytick.color": config.text_color,

        "font.family": config.font_family,
        "font.size": config.font_size_tick,

        "legend.fontsize": config.font_size_legend,
        "legend.frameon": theme == "publication",
        "legend.edgecolor": config.axes_edgecolor,
        "legend.facecolor": config.figure_facecolor,

        "lines.linewidth": config.linewidth_lines,
        "lines.markersize": 6,

        "savefig.dpi": config.dpi_save,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "savefig.facecolor": config.figure_facecolor,
        "savefig.edgecolor": config.figure_facecolor,

        "text.color": config.text_color,
        "image.cmap": "viridis",
    }

    if config.font_family == "serif":
        params["font.serif"] = font_stack
    else:
        params["font.sans-serif"] = font_stack

    plt.rcParams.update(params)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=config.color_palette)


def get_current_theme() -> Optional[str]:
    """Get the currently applied theme name."""
    return _current_theme


def get_plot_colors(theme: Optional[str] = None) -> Dict[str, str]:
    """Get named color roles for the current or specified theme."""
    theme = theme or _current_theme or "publication"
    config = THEMES[theme]
    palette = config.color_palette
    return {
        "primary": palette[0],
        "secondary": palette[1],
        "tertiary": palette[2],
        "accent": palette[5],
        "scatter": palette[1],
        "line": palette[0],
        "error": palette[5],
        "histogram": palette[1],
        "contour": "white" if theme == "dark" else "black",
        "text": config.text_color,
    }
