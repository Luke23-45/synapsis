"""
Shared experiment helpers for SYNAPSE empirical validation suite.

Provides:
  - Report creation and finalization
  - Run capsule setup with file logging
  - Publication-quality plotting (grouped bars, line, multi-panel)
  - CSV/JSONL recording
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from experiments.common.chart_exporter import save_chart
from experiments.common.report import ExperimentReport, TestCase, save_report_json
from experiments.common.theme import apply_theme
from experiments.utils.exporter import append_metrics_jsonl, save_results_csv
from experiments.utils.model_io import RunCapsule, create_run_capsule, save_config_snapshot, save_run_pointer

from .config import to_plain_dict

log = logging.getLogger(__name__)

# Human-readable baseline names
BASELINE_LABELS: Dict[str, str] = {
    "B0": "FixedWindow",
    "B1": "GRU",
    "B2": "Transformer",
    "B3": "UniformSample",
    "B4": "AnchorOnly",
    "B5": "Anchor+Proxy",
    "B6": "SYNAPSE",
}


def baseline_label(code: str) -> str:
    """Map B0-B6 to human-readable label."""
    return BASELINE_LABELS.get(code, code)


def start_report(
    experiment_id: str,
    experiment_name: str,
    formal_reference: str,
    claim: str,
) -> ExperimentReport:
    return ExperimentReport(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        formal_reference=formal_reference,
        claim=claim,
    )


def setup_run(cfg, experiment_name: str) -> RunCapsule:
    """Create run capsule with config snapshot and file logging."""
    capsule = create_run_capsule(cfg.output_dir, experiment_name)
    save_config_snapshot(capsule, to_plain_dict(cfg))

    # Attach file handler for this run
    log_file = capsule.logs / "run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)
    log.info("Run capsule: %s", capsule.root)
    log.info("Log file:    %s", log_file)

    return capsule


def record_case(
    report: ExperimentReport,
    capsule: RunCapsule,
    csv_rows: List[dict],
    case_name: str,
    passed: bool,
    row: dict,
    error: str | None = None,
) -> None:
    report.add_case(TestCase(name=case_name, passed=passed, details=row, error=error))
    csv_rows.append(row)
    append_metrics_jsonl(row, capsule.metrics / "metrics.jsonl")


def finalize_and_save(
    report: ExperimentReport,
    capsule: RunCapsule,
    csv_rows: List[dict],
    csv_fields: List[str],
    cfg,
) -> ExperimentReport:
    report.finalize()
    save_results_csv(csv_rows, csv_fields, capsule.metrics / "results.csv")
    save_report_json(report, str(capsule.artifacts / "report.json"))
    save_run_pointer(capsule, cfg.output_dir)
    log.info("[%s] %s  (%d/%d passed)", report.experiment_id, report.status,
             report.passed_cases, report.total_cases)
    return report


# ---------------------------------------------------------------------------
# Publication-quality plotting
# ---------------------------------------------------------------------------

def plot_bar(
    names: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    save_path: Path,
    formats: Sequence[str],
    theme: str,
    *,
    errors: Sequence[float] | None = None,
    highlight_best: bool = True,
    color_map: Dict[str, str] | None = None,
) -> None:
    """Single bar chart with optional error bars and best-value highlight."""
    apply_theme(theme)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))

    colors = []
    for n in names:
        if color_map and n in color_map:
            colors.append(color_map[n])
        elif "SYNAPSE" in str(n).upper():
            colors.append("#2196F3")
        else:
            colors.append("#78909C")

    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5,
                  yerr=errors, capsize=4, error_kw={"linewidth": 1.2})

    if highlight_best and values:
        best_idx = int(np.argmax(values))
        bars[best_idx].set_edgecolor("#FF5722")
        bars[best_idx].set_linewidth(2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_chart(fig, save_path, list(formats), close_fig=True)


def plot_grouped_bar(
    group_names: Sequence[str],
    method_names: Sequence[str],
    values: Dict[str, List[float]],
    title: str,
    ylabel: str,
    save_path: Path,
    formats: Sequence[str],
    theme: str,
    *,
    errors: Dict[str, List[float]] | None = None,
) -> None:
    """Grouped bar chart: one cluster per group, one bar per method."""
    apply_theme(theme)
    n_groups = len(group_names)
    n_methods = len(method_names)
    width = 0.8 / n_methods
    x = np.arange(n_groups)

    method_colors = {
        "SYNAPSE": "#2196F3", "FixedWindow": "#78909C", "GRU": "#FFA726",
        "Transformer": "#AB47BC", "UniformSample": "#66BB6A",
        "AnchorOnly": "#EF5350", "Anchor+Proxy": "#26C6DA",
    }

    fig, ax = plt.subplots(figsize=(max(9, n_groups * 1.5), 5.5))
    for i, method in enumerate(method_names):
        vals = values[method]
        errs = errors[method] if errors else None
        offset = (i - n_methods / 2 + 0.5) * width
        color = method_colors.get(method, f"C{i}")
        ax.bar(x + offset, vals, width * 0.9, label=method, color=color,
               edgecolor="white", linewidth=0.5,
               yerr=errs, capsize=3, error_kw={"linewidth": 1.0})

    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=25, ha="right", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_chart(fig, save_path, list(formats), close_fig=True)


def plot_line(
    x: Sequence[float],
    series: Sequence[Sequence[float]],
    labels: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path,
    formats: Sequence[str],
    theme: str,
    *,
    markers: bool = True,
) -> None:
    """Multi-series line chart."""
    apply_theme(theme)
    fig, ax = plt.subplots(figsize=(9, 5))

    line_styles = ["-", "--", "-.", ":"]
    marker_styles = ["o", "s", "^", "D", "v", "p", "*"]

    for i, (vals, label) in enumerate(zip(series, labels)):
        style = line_styles[i % len(line_styles)]
        marker = marker_styles[i % len(marker_styles)] if markers else None
        lw = 2.5 if "SYNAPSE" in label.upper() else 1.5
        ax.plot(x, vals, linestyle=style, marker=marker, markersize=5,
                linewidth=lw, label=label)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_chart(fig, save_path, list(formats), close_fig=True)


def plot_heatmap(
    data: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    save_path: Path,
    formats: Sequence[str],
    theme: str,
    *,
    cmap: str = "YlOrRd_r",
    fmt: str = ".3f",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Annotated heatmap for parameter sweeps."""
    apply_theme(theme)
    fig, ax = plt.subplots(figsize=(max(7, len(col_labels) * 1.2),
                                    max(5, len(row_labels) * 0.8)))
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{data[i, j]:{fmt}}", ha="center", va="center",
                    fontsize=8, color="white" if data[i, j] < (vmax or data.max()) * 0.5 else "black")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    save_chart(fig, save_path, list(formats), close_fig=True)
