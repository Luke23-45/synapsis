"""
Per-run file-based logging setup.

Adapted from GibbsQ ``gibbsq.utils.logging``.

Every experiment run creates a ``logs/run.log`` file inside its capsule
directory.  The logger hierarchy means every ``logging.getLogger(...)``
call in the codebase routes through the root logger and into this file.

Usage
-----
::

    from experiments.utils.logging import setup_run_logging

    capsule = create_run_capsule(cfg.output_dir, "causality")
    setup_run_logging(capsule, verbose=True)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from experiments.utils.model_io import RunCapsule

__all__ = [
    "setup_run_logging",
    "get_run_logger",
]

# Track file handlers to avoid duplicate attachment
_attached_log_files: set[str] = set()

LOG_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_run_logging(
    capsule: RunCapsule,
    verbose: bool = False,
    experiment_id: str = "",
) -> logging.Logger:
    """
    Set up file-based and console logging for an experiment run.

    Creates ``<capsule>/logs/run.log`` and attaches a ``FileHandler``
    to the root logger so all downstream ``log.info(...)`` calls
    are captured.

    Parameters
    ----------
    capsule : RunCapsule
        The run capsule (logs dir will be used).
    verbose : bool
        If True, set console level to DEBUG.  Otherwise INFO.
    experiment_id : str
        Label printed in the banner line.

    Returns
    -------
    logging.Logger
        A logger named for the experiment.
    """
    root = logging.getLogger()
    level = logging.DEBUG if verbose else logging.INFO

    # Console handler (only add once)
    if not any(isinstance(h, logging.StreamHandler) and h.stream in (sys.stdout, sys.stderr)
               for h in root.handlers):
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(level)
        console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        root.addHandler(console)

    root.setLevel(logging.DEBUG)  # let file handler capture everything

    # File handler into capsule logs dir
    log_file = capsule.logs / "run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(log_file.resolve())

    if resolved not in _attached_log_files:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        root.addHandler(fh)
        _attached_log_files.add(resolved)

    logger = logging.getLogger(experiment_id or "synapse.experiment")
    logger.info("=" * 60)
    logger.info("Run started: %s", datetime.now().isoformat(sep=" ", timespec="seconds"))
    logger.info("Capsule:     %s", capsule.root)
    logger.info("Log file:    %s", log_file)
    logger.info("=" * 60)

    return logger


def get_run_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``synapse.experiment`` namespace."""
    return logging.getLogger(f"synapse.experiment.{name}")
