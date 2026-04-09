"""
Applied Compression Retention
===============================

Paper claim
    SYNAPSE compression preserves more task-relevant structure
    than naive compression on real robotics trajectories.

Methodology
    Compress each episode to a fixed anchor budget using:
      - Uniform subsampling
      - Delta-threshold anchors
      - SYNAPSE memory anchors
    Measure retained coverage of phase boundaries and events.

Metrics
    - retained_phase_coverage:  fraction of phase boundaries
                                captured within tolerance
    - retained_event_coverage:  fraction of large-delta events
                                captured
    - budget_efficiency:        anchors used / budget K
    - compression_ratio:        K / T

Outputs
    figures/
        compression_coverage.{png,pdf}
        compression_budget_efficiency.{png,pdf}
    metrics/results.csv, metrics/metrics.jsonl
    artifacts/report.json
    logs/run.log
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from time import perf_counter
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.empirical.common.baselines import synapse_feature
from experiments.empirical.common.data import (
    RobotEpisode, load_applied_dataset, generate_synthetic_episodes,
)
from experiments.empirical.common.experiment import (
    finalize_and_save, plot_grouped_bar, record_case, setup_run, start_report,
)
from experiments.empirical.common.config import load_config, validate_config

log = logging.getLogger(__name__)

EXPERIMENT_ID = "AP-02"
EXPERIMENT_NAME = "Compression Retention"

CSV_FIELDS = [
    "episode_id", "method", "budget", "anchors_used",
    "retained_phase_coverage", "retained_event_coverage",
    "budget_efficiency", "compression_ratio", "passed",
]

METHODS = ["UniformSample", "DeltaThreshold", "SYNAPSE"]


def _get_anchors(seq: np.ndarray, method: str, K: int, r: int,
                 tau: float, Q: int, weights: tuple) -> List[int]:
    """Return anchor indices for a method."""
    if method == "UniformSample":
        n = min(K, len(seq))
        return np.linspace(0, len(seq) - 1, num=n, dtype=int).tolist()

    if method == "DeltaThreshold":
        diffs = np.zeros(len(seq), dtype=np.float32)
        if len(seq) > 1:
            diffs[1:] = np.linalg.norm(np.diff(seq, axis=0), axis=1)
        return np.where(diffs > tau)[0].tolist()[:K]

    _, anchor_indices, _ = synapse_feature(seq, K, r, tau, Q, weights)
    return anchor_indices


def _coverage(anchors: List[int], targets: List[int], tolerance: int) -> float:
    """Fraction of targets hit by at least one anchor."""
    if not targets:
        return 1.0
    return sum(1 for t in targets
               if any(abs(a - t) <= tolerance for a in anchors)) / len(targets)


def _detect_events(seq: np.ndarray, quantile: float = 0.9) -> List[int]:
    """Detect large-delta events in a sequence."""
    if len(seq) < 2:
        return []
    diffs = np.linalg.norm(np.diff(seq, axis=0), axis=1)
    threshold = np.quantile(diffs, quantile)
    return np.where(diffs > threshold)[0].tolist()


def _load_dataset(cfg):
    export_root = getattr(cfg.applied_data, "export_root", "")
    index_json = getattr(cfg.applied_data, "index_json", "")
    if export_root and Path(export_root).exists():
        return load_applied_dataset(export_root, index_json)
    log.warning("Real data not found. Using synthetic fallback.")
    return generate_synthetic_episodes(num_episodes=10, T=100)


def run_experiment(cfg=None, verbose: bool = False):
    if cfg is None:
        cfg = load_config("experiments/configs/robotics.yaml")
        validate_config(cfg)

    report = start_report(
        EXPERIMENT_ID, EXPERIMENT_NAME,
        "Paper claim: SYNAPSE compression preserves task-relevant structure",
        "SYNAPSE retains more phase boundaries and events than baselines",
    )
    capsule = setup_run(cfg, "applied_compression_retention")
    csv_rows: list[dict] = []
    start = perf_counter()

    dataset = _load_dataset(cfg)
    tolerance = cfg.applied_data.boundary_tolerance

    K = cfg.memory.K
    r = cfg.memory.r
    tau = cfg.memory.tau
    Q = cfg.memory.Q
    weights = tuple(cfg.memory.weights)

    method_phase_cov: dict[str, list[float]] = {m: [] for m in METHODS}
    method_event_cov: dict[str, list[float]] = {m: [] for m in METHODS}
    method_efficiency: dict[str, list[float]] = {m: [] for m in METHODS}

    for ep in dataset.episodes:
        if ep.length < cfg.applied_data.min_episode_length:
            continue

        seq = ep.state_sequence
        boundaries = ep.expert_state_boundaries
        events = _detect_events(seq)

        for method in METHODS:
            anchors = _get_anchors(seq, method, K, r, tau, Q, weights)
            phase_cov = _coverage(anchors, boundaries, tolerance)
            event_cov = _coverage(anchors, events, tolerance)
            eff = len(anchors) / max(1, K)
            comp_ratio = K / max(1, ep.length)

            method_phase_cov[method].append(phase_cov)
            method_event_cov[method].append(event_cov)
            method_efficiency[method].append(eff)

            row = {
                "episode_id": ep.episode_id,
                "method": method,
                "budget": K,
                "anchors_used": len(anchors),
                "retained_phase_coverage": round(phase_cov, 6),
                "retained_event_coverage": round(event_cov, 6),
                "budget_efficiency": round(eff, 4),
                "compression_ratio": round(comp_ratio, 4),
                "passed": True,
            }
            record_case(report, capsule, csv_rows,
                         f"{ep.episode_id}_{method}", True, row)

    report.duration_seconds = perf_counter() - start

    # ---- Publication figures -----------------------------------------------
    metric_names = ["Phase Coverage", "Event Coverage", "Budget Efficiency"]
    grouped_vals: dict[str, list[float]] = {}
    grouped_errs: dict[str, list[float]] = {}

    for m in METHODS:
        grouped_vals[m] = [
            float(np.mean(method_phase_cov[m])) if method_phase_cov[m] else 0.0,
            float(np.mean(method_event_cov[m])) if method_event_cov[m] else 0.0,
            float(np.mean(method_efficiency[m])) if method_efficiency[m] else 0.0,
        ]
        grouped_errs[m] = [
            float(np.std(method_phase_cov[m])) if method_phase_cov[m] else 0.0,
            float(np.std(method_event_cov[m])) if method_event_cov[m] else 0.0,
            float(np.std(method_efficiency[m])) if method_efficiency[m] else 0.0,
        ]

    plot_grouped_bar(
        metric_names, METHODS, grouped_vals,
        "Compression Retention: Phase & Event Coverage",
        "Coverage / Efficiency",
        capsule.figures / "compression_coverage",
        cfg.plotting.formats, cfg.plotting.theme,
        errors=grouped_errs,
    )

    # Acceptance
    syn_cov = float(np.mean(method_phase_cov["SYNAPSE"])) if method_phase_cov["SYNAPSE"] else 0.0
    gate = cfg.acceptance_gates.applied_compression_coverage
    report.metadata["acceptance_passed"] = bool(syn_cov >= gate)
    report.metadata["mean_phase_coverage"] = {
        m: round(float(np.mean(method_phase_cov[m])), 6)
        if method_phase_cov[m] else 0.0
        for m in METHODS
    }
    log.info("AP-02 SYNAPSE phase coverage: %.4f (gate: %.4f)", syn_cov, gate)

    return finalize_and_save(report, capsule, csv_rows, CSV_FIELDS, cfg)


if __name__ == "__main__":
    config_path = "experiments/configs/robotics.yaml"
    for arg in sys.argv:
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
    cfg = load_config(config_path)
    validate_config(cfg)
    v = "--verbose" in sys.argv or "-v" in sys.argv
    report = run_experiment(cfg=cfg, verbose=v)
    print(f"[{report.status}] {EXPERIMENT_ID}: {EXPERIMENT_NAME}")
    sys.exit(0 if report.status == "PASS" else 1)
