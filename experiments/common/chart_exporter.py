"""
Unified chart and data export system for SYNAPSE experiments.

Adapted from GibbsQ's chart_exporter.py — provides:
- Multi-format chart export (PNG, PDF, SVG)
- High-DPI output for print quality
- Data export (CSV, JSON, NPZ)
- Consistent naming conventions
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mfig

log = logging.getLogger(__name__)

__all__ = [
    "save_chart",
    "save_data",
    "save_chart_with_data",
    "ChartConfig",
    "DataConfig",
    "export_experiment_results",
]


@dataclass
class ChartConfig:
    """Configuration for chart export."""
    dpi: int = 600
    transparent: bool = False
    bbox_inches: str = "tight"
    pad_inches: float = 0.1
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {
                "Creator": "SYNAPSE Chart Exporter",
            }


@dataclass
class DataConfig:
    """Configuration for data export."""
    indent: int = 2
    compress: bool = False
    include_index: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


def save_chart(
    fig: mfig.Figure,
    output_path: Union[str, Path],
    formats: Optional[List[str]] = None,
    config: Optional[ChartConfig] = None,
    close_fig: bool = True,
) -> List[Path]:
    """
    Save a matplotlib figure in multiple formats.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save.
    output_path : str or Path
        Base path (without extension).
    formats : list of str
        Formats: ['png', 'pdf', 'svg']. Default: ['png'].
    config : ChartConfig
    close_fig : bool
        Whether to close the figure after saving.

    Returns
    -------
    list of Path
        Paths to saved files.
    """
    if formats is None:
        formats = ["png"]
    if config is None:
        config = ChartConfig()

    output_path = Path(output_path)
    output_dir = output_path.parent
    base_name = output_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        file_path = output_dir / f"{base_name}.{fmt}"

        try:
            fig.savefig(
                file_path,
                format=fmt,
                dpi=config.dpi,
                transparent=config.transparent,
                bbox_inches=config.bbox_inches,
                pad_inches=config.pad_inches,
                metadata=config.metadata if fmt == "pdf" else None,
            )
            saved_paths.append(file_path)
            log.debug(f"Saved chart: {file_path}")
        except Exception as e:
            log.error(f"Failed to save {fmt} chart: {e}")

    if close_fig:
        plt.close(fig)

    return saved_paths


def save_data(
    data: Dict[str, Any],
    output_path: Union[str, Path],
    format: str = "json",
    config: Optional[DataConfig] = None,
) -> Path:
    """
    Save data in specified format.

    Parameters
    ----------
    data : dict
    output_path : str or Path
        Base path (without extension).
    format : str
        'json', 'csv', or 'npz'.
    config : DataConfig

    Returns
    -------
    Path
    """
    if config is None:
        config = DataConfig()

    output_path = Path(output_path)
    format = format.lower().lstrip(".")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_path = output_path.with_suffix(f".{format}")

    if format == "json":
        _save_json(data, file_path, config)
    elif format == "csv":
        _save_csv(data, file_path, config)
    elif format == "npz":
        _save_npz(data, file_path, config)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json', 'csv', or 'npz'")

    log.debug(f"Saved data: {file_path}")
    return file_path


def save_chart_with_data(
    fig: mfig.Figure,
    data: Dict[str, Any],
    output_path: Union[str, Path],
    chart_formats: Optional[List[str]] = None,
    data_format: str = "json",
    chart_config: Optional[ChartConfig] = None,
    data_config: Optional[DataConfig] = None,
) -> Dict[str, List[Path]]:
    """
    Save both chart and associated data side by side.

    Returns
    -------
    dict with 'charts' and 'data' keys.
    """
    output_path = Path(output_path)

    chart_paths = save_chart(fig, output_path, chart_formats, chart_config, close_fig=False)
    data_path = output_path.parent / f"{output_path.stem}_data"
    data_paths = [save_data(data, data_path, data_format, data_config)]

    plt.close(fig)

    return {"charts": chart_paths, "data": data_paths}


def export_experiment_results(
    output_dir: Union[str, Path],
    figures: Optional[Dict[str, mfig.Figure]] = None,
    data: Optional[Dict[str, Dict[str, Any]]] = None,
    chart_formats: Optional[List[str]] = None,
    data_format: str = "json",
    chart_config: Optional[ChartConfig] = None,
    data_config: Optional[DataConfig] = None,
) -> Dict[str, List[Path]]:
    """
    Export all experiment results (multiple figures and data).

    Parameters
    ----------
    output_dir : str or Path
    figures : dict mapping name → Figure
    data : dict mapping name → data dict
    chart_formats : list of str
    data_format : str
    chart_config : ChartConfig
    data_config : DataConfig

    Returns
    -------
    dict with 'charts' and 'data' keys.
    """
    if chart_formats is None:
        chart_formats = ["png", "pdf"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_chart_paths = []
    all_data_paths = []

    if figures:
        for name, fig in figures.items():
            chart_path = output_dir / name
            paths = save_chart(fig, chart_path, chart_formats, chart_config, close_fig=False)
            all_chart_paths.extend(paths)
            plt.close(fig)

    if data:
        for name, data_dict in data.items():
            data_path = output_dir / f"{name}_data"
            path = save_data(data_dict, data_path, data_format, data_config)
            all_data_paths.append(path)

    return {"charts": all_chart_paths, "data": all_data_paths}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _save_json(data: Dict[str, Any], file_path: Path, config: DataConfig) -> None:
    """Save data as JSON."""
    serializable = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serializable[key] = float(value)
        elif isinstance(value, (np.bool_,)):
            serializable[key] = bool(value)
        else:
            serializable[key] = value

    if config.metadata:
        serializable["_metadata"] = config.metadata

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=config.indent, default=str)


def _save_csv(data: Dict[str, Any], file_path: Path, config: DataConfig) -> None:
    """Save data as CSV."""
    try:
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=config.include_index)
    except ImportError:
        # Fallback: manual CSV
        import csv
        keys = list(data.keys())
        max_len = max(
            (len(v) if isinstance(v, (list, np.ndarray)) else 1)
            for v in data.values()
        )
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for i in range(max_len):
                row = []
                for k in keys:
                    v = data[k]
                    if isinstance(v, (list, np.ndarray)) and i < len(v):
                        row.append(v[i])
                    elif not isinstance(v, (list, np.ndarray)):
                        row.append(v if i == 0 else "")
                    else:
                        row.append("")
                writer.writerow(row)


def _save_npz(data: Dict[str, Any], file_path: Path, config: DataConfig) -> None:
    """Save data as NPZ."""
    np_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            np_data[key] = value
        elif isinstance(value, (list, tuple)):
            np_data[key] = np.array(value)
        else:
            np_data[key] = np.array(value)

    if config.compress:
        np.savez_compressed(file_path, **np_data)
    else:
        np.savez(file_path, **np_data)
