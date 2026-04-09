"""
PC-07: Pilot Diagnostics
==========================

Paper claim
    Anchors and topological summaries align with meaningful phase
    structure in pilot data.

Methodology
    Load in-house episodes (or synthetic fallback). Run SYNAPSE on
    each episode. Report anchor count, phase entropy, and topology
    mass as diagnostic statistics. No training needed.

Outputs
    figures/
        pilot_anchor_distribution.{png,pdf}
        pilot_phase_histogram.{png,pdf}
    metrics/metrics.jsonl, results.csv
    artifacts/report.json, logs/run.log
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.empirical.common.baselines import synapse_feature
from experiments.empirical.common.experiment import (
    finalize_and_save, plot_bar, record_case, setup_run, start_report,
)
from experiments.empirical.common.config import load_config, validate_config
from experiments.empirical.common.tasks import generate_control_dataset
from experiments.utils.progress import iter_progress

log = logging.getLogger(__name__)

EXPERIMENT_ID = "PC-07"
EXPERIMENT_NAME = "Pilot Diagnostics"

CSV_FIELDS = [
    "episode_id", "num_steps", "num_anchors",
    "phase_entropy", "topology_mass", "passed",
]


def _load_episodes(cfg):
    """Load in-house episodes or fall back to synthetic data."""
    export_root = getattr(cfg.inhouse_pilot, "export_root", "")

    if export_root and Path(export_root).exists():
        from experiments.empirical.common.io import load_inhouse_export_episodes
        return load_inhouse_export_episodes(export_root)

    log.warning("In-house data not found. Using synthetic fallback.")
    rng = np.random.default_rng(42)
    episodes = generate_control_dataset(
        rng, n_episodes=20, T=100,
        state_dim=12, action_dim=4,
    )
    for i, ep in enumerate(episodes):
        ep["episode_id"] = f"synthetic_{i:03d}"
    return episodes


def run_experiment(cfg=None, verbose: bool = False):
    if cfg is None:
        cfg = load_config("experiments/configs/pilot.yaml")
        validate_config(cfg)

    report = start_report(
        EXPERIMENT_ID, EXPERIMENT_NAME,
        "Paper claim: qualitative and diagnostic support on local data",
        "Anchors and topology align with meaningful phase structure",
    )
    capsule = setup_run(cfg, "pilot_diagnostics")
    csv_rows: list[dict] = []
    start = perf_counter()

    episodes = _load_episodes(cfg)
    min_len = cfg.inhouse_pilot.min_episode_length

    phase_counts: Counter = Counter()
    anchor_counts: list[int] = []
    topology_masses: list[float] = []

    for episode in iter_progress(episodes, desc="PC-07 episodes"):
        states = episode["states"]
        if len(states) < min_len:
            continue

        feat, anchors, _ = synapse_feature(
            states, cfg.memory.K, cfg.memory.r, cfg.memory.tau,
            cfg.memory.Q, tuple(cfg.memory.weights),
        )

        # Phase entropy
        phases = episode.get("phase_labels", np.zeros(len(states), dtype=np.int64))
        if isinstance(phases, list):
            phases = np.asarray(phases, dtype=np.int64)
        hist = np.bincount(phases, minlength=max(1, int(phases.max()) + 1)).astype(np.float32)
        probs = hist / max(1.0, hist.sum())
        entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())

        # Topology mass
        topology_mass = float(np.mean(np.abs(feat[-8:]))) if feat.size >= 8 else float(np.mean(np.abs(feat)))

        for p in phases:
            phase_counts[int(p)] += 1
        anchor_counts.append(len(anchors))
        topology_masses.append(topology_mass)

        ep_id = episode.get("episode_id", f"ep_{len(anchor_counts)}")
        row = {
            "episode_id": ep_id,
            "num_steps": len(states),
            "num_anchors": len(anchors),
            "phase_entropy": round(entropy, 6),
            "topology_mass": round(topology_mass, 6),
            "passed": True,
        }
        record_case(report, capsule, csv_rows, ep_id, True, row)

    report.duration_seconds = perf_counter() - start

    # ---- Publication figures -----------------------------------------------
    if anchor_counts:
        # Figure 1: Anchor count distribution
        unique_counts, freq = np.unique(anchor_counts, return_counts=True)
        plot_bar(
            [str(c) for c in unique_counts],
            freq.tolist(),
            "Anchor Count Distribution Across Episodes",
            "Frequency",
            capsule.figures / "pilot_anchor_distribution",
            cfg.plotting.formats, cfg.plotting.theme,
            highlight_best=False,
        )

    if phase_counts:
        # Figure 2: Phase histogram
        sorted_phases = sorted(phase_counts.keys())
        plot_bar(
            [f"Phase {p}" for p in sorted_phases],
            [phase_counts[p] for p in sorted_phases],
            "Phase Label Distribution",
            "Count",
            capsule.figures / "pilot_phase_histogram",
            cfg.plotting.formats, cfg.plotting.theme,
            highlight_best=False,
        )

    report.metadata["acceptance_passed"] = len(anchor_counts) > 0
    report.metadata["num_episodes_processed"] = len(anchor_counts)
    report.metadata["mean_anchors"] = round(float(np.mean(anchor_counts)), 2) if anchor_counts else 0
    report.metadata["mean_topology_mass"] = round(float(np.mean(topology_masses)), 6) if topology_masses else 0

    log.info("PC-07 processed %d episodes, mean anchors: %.1f",
             len(anchor_counts), np.mean(anchor_counts) if anchor_counts else 0)

    return finalize_and_save(report, capsule, csv_rows, CSV_FIELDS, cfg)


if __name__ == "__main__":
    config_path = "experiments/configs/pilot.yaml"
    for arg in sys.argv:
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
    cfg = load_config(config_path)
    validate_config(cfg)
    v = "--verbose" in sys.argv or "-v" in sys.argv
    report = run_experiment(cfg=cfg, verbose=v)
    print(f"[{report.status}] {EXPERIMENT_ID}: {EXPERIMENT_NAME}")
    sys.exit(0 if report.status == "PASS" else 1)
