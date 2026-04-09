"""
Data export utilities (JSONL, CSV).

Adapted from GibbsQ ``gibbsq.utils.exporter``.

JSONL (.jsonl)
    One JSON object per line.  Used for appending per-case metrics
    so each experiment case lands on its own line.

CSV (.csv)
    Flat table with header.  Used for the final results summary
    that goes directly into publication tables.

Usage
-----
::

    from experiments.utils.exporter import append_metrics_jsonl, save_results_csv

    append_metrics_jsonl({"case": "K10_r2", "passed": True, "max_m": 7}, metrics_path)
    save_results_csv(rows, fieldnames, csv_path)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "append_metrics_jsonl",
    "save_results_csv",
]


def _clean_value(v: Any) -> Any:
    """Convert numpy scalars/arrays to JSON-safe Python types."""
    if isinstance(v, (np.generic, np.ndarray)):
        return v.tolist()
    return v


def append_metrics_jsonl(
    metrics: Dict[str, Any],
    file_path: str | Path,
) -> None:
    """
    Append a single metric record to a JSON Lines file.

    Parameters
    ----------
    metrics : dict
        Scalar key-value pairs for one test case.
    file_path : str or Path
        Path to the ``.jsonl`` file (created if not exists).
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clean = {k: _clean_value(v) for k, v in metrics.items()}

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(clean) + "\n")


def save_results_csv(
    rows: Sequence[Dict[str, Any]],
    fieldnames: List[str],
    file_path: str | Path,
) -> None:
    """
    Save structured results to a CSV file with header.

    Parameters
    ----------
    rows : list of dict
        Each dict is one row.  Keys must be a superset of *fieldnames*.
    fieldnames : list of str
        Column order in the output CSV.
    file_path : str or Path
        Destination CSV path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean_row = {k: _clean_value(v) for k, v in row.items()}
            writer.writerow(clean_row)

    log.debug("Saved CSV (%d rows) -> %s", len(rows), path)
