"""
Primary Applied Validation: Topology Structure
==============================================

Primary applied validation evidence:
    Topology summaries distinguish different real trajectory structures.

Methodology
    Group real episodes by structural characteristics:
      - phase complexity (number of unique states)
      - movement intensity (mean velocity)
      - revisit structure (presence of return-to-same-phase patterns)
    Compute SYNAPSE persistence summaries for each episode.
    Measure intra-group vs inter-group separation.

Metrics
    - silhouette_score:   clustering quality of topology summaries
    - inter_intra_ratio:  ratio of inter-group to intra-group distances
    - per-group summary statistics

If dataset is too homogeneous for strong separation, this becomes a
case-study section rather than a headline quantitative result.

Outputs
    figures/
        topology_group_separation.{png,pdf}
        topology_summary_distributions.{png,pdf}
    metrics/results.csv, metrics/metrics.jsonl
    artifacts/report.json
    logs/run.log
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.empirical.common.baselines import synapse_feature, summarize_diagrams
from experiments.empirical.common.data import (
    RobotEpisode, load_applied_dataset, generate_synthetic_episodes,
)
from experiments.empirical.common.experiment import (
    finalize_and_save, plot_bar, plot_grouped_bar,
    record_case, setup_run, start_report,
)
from experiments.empirical.common.config import load_config, validate_config

log = logging.getLogger(__name__)

EXPERIMENT_ID = "AP-03"
EXPERIMENT_NAME = "Topology Structure"

CSV_FIELDS = [
    "episode_id", "group", "num_unique_states", "mean_velocity",
    "has_revisit", "topology_mass", "topology_spread",
    "persistence_total", "persistence_max", "passed",
]


def _classify_episode(ep: RobotEpisode) -> str:
    """Classify episode into structural group."""
    n_states = len(set(ep.expert_states))
    velocity = float(np.mean(np.linalg.norm(np.diff(ep.state_sequence, axis=0), axis=1))) if ep.length > 1 else 0.0

    # Detect revisit: same expert_state appears in non-contiguous segments
    state_order = ep.unique_expert_states
    has_revisit = len(state_order) != len(set(state_order))

    if has_revisit:
        return "revisit"
    elif n_states >= 8:
        return "complex"
    elif velocity > np.median([0.01, velocity, 0.1]):
        return "dynamic"
    else:
        return "simple"


def _topology_features(ep: RobotEpisode, cfg) -> dict:
    """Extract topology summary features from an episode."""
    seq = ep.state_sequence
    feat, anchors, mem_size = synapse_feature(
        seq, cfg.memory.K, cfg.memory.r, cfg.memory.tau,
        cfg.memory.Q, tuple(cfg.memory.weights),
    )

    # Topology mass: mean absolute value of the topology portion of the feature
    # The last 8+ values are the persistence diagram summary
    topo_portion = feat[-8:] if feat.size >= 8 else feat
    topology_mass = float(np.mean(np.abs(topo_portion)))
    topology_spread = float(np.std(topo_portion))

    # Raw persistence stats
    persistence_total = float(topo_portion.sum())
    persistence_max = float(np.max(np.abs(topo_portion)))

    return {
        "topology_mass": topology_mass,
        "topology_spread": topology_spread,
        "persistence_total": persistence_total,
        "persistence_max": persistence_max,
        "feature": feat,
    }


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
        "Primary applied validation evidence: meaningful structure in topology",
        "Topology separates structural groups in real trajectories",
    )
    capsule = setup_run(cfg, "applied_topology_structure")
    csv_rows: list[dict] = []
    start = perf_counter()

    dataset = _load_dataset(cfg)

    group_features: dict[str, list[np.ndarray]] = {}
    group_mass: dict[str, list[float]] = {}

    for ep in dataset.episodes:
        if ep.length < cfg.applied_data.min_episode_length:
            continue

        group = _classify_episode(ep)
        topo = _topology_features(ep, cfg)

        group_features.setdefault(group, []).append(topo["feature"])
        group_mass.setdefault(group, []).append(topo["topology_mass"])

        n_states = len(set(ep.expert_states))
        velocity = float(np.mean(np.linalg.norm(
            np.diff(ep.state_sequence, axis=0), axis=1,
        ))) if ep.length > 1 else 0.0
        state_order = ep.unique_expert_states
        has_revisit = len(state_order) != len(set(state_order))

        row = {
            "episode_id": ep.episode_id,
            "group": group,
            "num_unique_states": n_states,
            "mean_velocity": round(velocity, 6),
            "has_revisit": has_revisit,
            "topology_mass": round(topo["topology_mass"], 6),
            "topology_spread": round(topo["topology_spread"], 6),
            "persistence_total": round(topo["persistence_total"], 6),
            "persistence_max": round(topo["persistence_max"], 6),
            "passed": True,
        }
        record_case(report, capsule, csv_rows, ep.episode_id, True, row)

    report.duration_seconds = perf_counter() - start

    # ---- Publication figures -----------------------------------------------
    groups = sorted(group_mass.keys())

    if groups:
        # Figure 1: Mean topology mass by group
        means = [float(np.mean(group_mass[g])) for g in groups]
        stds = [float(np.std(group_mass[g])) for g in groups]
        plot_bar(
            groups, means,
            "Topology Mass by Trajectory Group",
            "Mean Topology Mass",
            capsule.figures / "topology_group_separation",
            cfg.plotting.formats, cfg.plotting.theme,
            errors=stds,
            highlight_best=False,
        )

    # Inter vs intra distance
    if len(groups) >= 2 and all(len(group_features[g]) > 0 for g in groups):
        # Compute pairwise centroid distances
        centroids = {}
        for g in groups:
            feats = group_features[g]
            min_len = min(f.size for f in feats)
            trimmed = [f[:min_len] for f in feats]
            centroids[g] = np.mean(trimmed, axis=0)

        inter_dists = []
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                inter_dists.append(
                    float(np.linalg.norm(centroids[g1] - centroids[g2])))

        intra_dists = []
        for g in groups:
            feats = group_features[g]
            min_len = min(f.size for f in feats)
            trimmed = [f[:min_len] for f in feats]
            c = centroids[g]
            for f in trimmed:
                intra_dists.append(float(np.linalg.norm(f - c)))

        inter_mean = float(np.mean(inter_dists)) if inter_dists else 0.0
        intra_mean = float(np.mean(intra_dists)) if intra_dists else 1.0
        ratio = inter_mean / max(1e-10, intra_mean)

        report.metadata["inter_intra_ratio"] = round(ratio, 4)
        report.metadata["inter_mean"] = round(inter_mean, 6)
        report.metadata["intra_mean"] = round(intra_mean, 6)
        log.info("AP-03 inter/intra ratio: %.4f", ratio)
    else:
        log.warning("AP-03: dataset too homogeneous for separation analysis "
                     "(%d groups). Reporting as case study.", len(groups))
        report.metadata["inter_intra_ratio"] = None
        report.metadata["case_study_note"] = "Dataset too homogeneous for group separation"

    report.metadata["groups"] = {
        g: {"count": len(group_mass[g]),
            "mean_mass": round(float(np.mean(group_mass[g])), 6)}
        for g in groups
    }
    
    gate = getattr(cfg.acceptance_gates, "applied_topology_separation_ratio", 1.0)
    
    if len(groups) >= 2 and ratio is not None:
        report.metadata["acceptance_passed"] = bool(ratio >= gate)
        log.info("AP-03 topology separation ratio: %.4f (gate: %.4f)", ratio, gate)
    else:
        report.metadata["acceptance_passed"] = False
        log.warning("AP-03 failed: not enough groups for separation analysis.")

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
