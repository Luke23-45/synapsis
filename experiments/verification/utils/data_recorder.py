"""
Data Recorder for Z2 Verification Experiments.

Wraps the RunCapsule system to provide verification-specific helpers
for recording complete experiment data (not just pass/fail). This
enables retrospective visualization and publication-quality graphs
without re-running experiments.

Usage
-----
::

    recorder = VerificationRecorder(capsule, experiment_id="VZ2-01")
    recorder.log_scalar(trial=0, metric="optimality_gap", value=1e-6)
    recorder.save_trial_data("trial_000", scores=scores, y_star=y_star)
    recorder.save_sweep("lambda_sweep", lambdas=lambdas, l2_norms=norms)
    recorder.finalize()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from experiments.utils.model_io import RunCapsule, save_experiment_artifact

log = logging.getLogger(__name__)


class VerificationRecorder:
    """Records complete experiment data alongside pass/fail status.

    The recorder organizes data into three tiers:

    1. **Scalars** (JSONL) — per-trial metrics streamed to disk.
       Crash-safe, append-only.  Good for loss values, gap magnitudes,
       boolean pass/fail per trial, timing.

    2. **Trial Data** (NPZ) — per-trial dense arrays.
       Scores, selector outputs, projected indices, point clouds,
       persistence diagrams.  One file per trial or per parameter combo.

    3. **Sweeps** (NPZ) — aggregated arrays across a parameter axis.
       Lambda-vs-L2-norm curves, epsilon-vs-bottleneck traces,
       margin-vs-recovery-rate tables.

    All data lands inside the RunCapsule's directory tree under
    ``artifacts/data/`` so it coexists with (and does not disturb)
    the existing ``report.json`` and ``metrics.jsonl``.
    """

    def __init__(self, capsule: RunCapsule, experiment_id: str) -> None:
        self.capsule = capsule
        self.experiment_id = experiment_id

        # Data root lives under artifacts/data/
        self.data_root = capsule.artifacts / "data"
        self.data_root.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.trials_dir = self.data_root / "trials"
        self.trials_dir.mkdir(exist_ok=True)

        self.sweeps_dir = self.data_root / "sweeps"
        self.sweeps_dir.mkdir(exist_ok=True)

        # Streaming scalar log
        self._scalar_path = self.data_root / "scalars.jsonl"
        self._scalar_count = 0

        log.info("[Recorder] Initialized for %s → %s", experiment_id, self.data_root)

    # ------------------------------------------------------------------
    # Tier 1: Streaming scalars
    # ------------------------------------------------------------------

    def log_scalar(self, **kwargs: Any) -> None:
        """Append a single scalar record to the JSONL log.

        Parameters
        ----------
        **kwargs
            Arbitrary key-value pairs.  Numpy scalars are auto-converted.

        Example
        -------
        ::

            recorder.log_scalar(trial=0, d=5, T=100, lam=0.1,
                                gap=1.2e-7, feasible=True, passed=True)
        """
        record = {k: _to_json_safe(v) for k, v in kwargs.items()}
        with open(self._scalar_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")
        self._scalar_count += 1

    # ------------------------------------------------------------------
    # Tier 2: Per-trial dense data
    # ------------------------------------------------------------------

    def save_trial_data(self, trial_name: str, **arrays: np.ndarray) -> Path:
        """Save dense arrays for a single trial.

        Parameters
        ----------
        trial_name : str
            Unique name for this trial (e.g., ``"d5_T100_lam0.1"``).
        **arrays
            Named numpy arrays to persist.

        Returns
        -------
        Path
            Path to the saved ``.npz`` file.
        """
        path = self.trials_dir / f"{trial_name}.npz"
        np.savez_compressed(path, **arrays)
        log.debug("[Recorder] Trial data saved: %s", path.name)
        return path

    # ------------------------------------------------------------------
    # Tier 3: Parameter sweeps
    # ------------------------------------------------------------------

    def save_sweep(self, sweep_name: str, **arrays: np.ndarray) -> Path:
        """Save aggregated sweep data (e.g., lambda vs L2 norm curve).

        Parameters
        ----------
        sweep_name : str
            Descriptive name (e.g., ``"lambda_l2_shrinkage"``).
        **arrays
            Named numpy arrays — typically 1-D axes and their responses.

        Returns
        -------
        Path
            Path to the saved ``.npz`` file.
        """
        path = self.sweeps_dir / f"{sweep_name}.npz"
        np.savez_compressed(path, **arrays)
        log.debug("[Recorder] Sweep data saved: %s", path.name)
        return path

    # ------------------------------------------------------------------
    # Persistence diagrams (special case — list of variable-length arrays)
    # ------------------------------------------------------------------

    def save_diagrams(
        self,
        name: str,
        diagrams: List[np.ndarray],
        *,
        extra: Optional[Dict[str, np.ndarray]] = None,
    ) -> Path:
        """Save persistence diagrams alongside optional metadata arrays.

        Each diagram is stored as ``dgm_0``, ``dgm_1``, ... inside a
        single ``.npz`` file.  Extra arrays (e.g., the point cloud that
        produced them) are stored alongside.

        Parameters
        ----------
        name : str
            File stem.
        diagrams : list of np.ndarray
            One array per homology degree.
        extra : dict, optional
            Additional arrays to include.

        Returns
        -------
        Path
        """
        data: Dict[str, np.ndarray] = {}
        for i, dgm in enumerate(diagrams):
            data[f"dgm_{i}"] = np.asarray(dgm)
        data["num_diagrams"] = np.array(len(diagrams))
        if extra:
            data.update(extra)

        path = self.trials_dir / f"{name}.npz"
        np.savez_compressed(path, **data)
        log.debug("[Recorder] Diagrams saved: %s", path.name)
        return path

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self) -> Dict[str, Any]:
        """Write a summary manifest and return statistics.

        Returns
        -------
        dict
            Summary with counts of saved files.
        """
        trial_files = list(self.trials_dir.glob("*.npz"))
        sweep_files = list(self.sweeps_dir.glob("*.npz"))

        manifest = {
            "experiment_id": self.experiment_id,
            "scalar_records": self._scalar_count,
            "trial_files": len(trial_files),
            "sweep_files": len(sweep_files),
            "trial_names": sorted(f.stem for f in trial_files),
            "sweep_names": sorted(f.stem for f in sweep_files),
        }

        manifest_path = self.data_root / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        log.info(
            "[Recorder] Finalized %s: %d scalars, %d trials, %d sweeps",
            self.experiment_id,
            self._scalar_count,
            len(trial_files),
            len(sweep_files),
        )
        return manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_json_safe(v: Any) -> Any:
    """Convert numpy scalars to Python natives for JSON."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _json_default(obj: Any) -> Any:
    """Fallback serializer for json.dumps."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
