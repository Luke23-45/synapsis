"""
Primary Applied Validation: Dataset Audit
=========================================

First gate for all applied-data experiments. Verifies data integrity
before any SYNAPSE analysis runs.

Validates
    - episode counts match index
    - required columns present
    - phase labels exist and are finite
    - proprioception / action shapes correct
    - per-episode integrity

This is NOT a paper headline experiment. It is a trust prerequisite.

Outputs
    figures/audit_episode_lengths.{png,pdf}
    metrics/results.csv
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

from experiments.empirical.common.data import (
    PROPRIO_DIM, ACTION_DIM, NUM_PHASES,
    load_applied_dataset, validate_dataset, validate_episode,
    generate_synthetic_episodes,
)
from experiments.empirical.common.experiment import (
    finalize_and_save, plot_bar, record_case, setup_run, start_report,
)
from experiments.empirical.common.config import load_config, validate_config

log = logging.getLogger(__name__)

EXPERIMENT_ID = "AP-00"
EXPERIMENT_NAME = "Applied Dataset Audit"

CSV_FIELDS = [
    "episode_id", "length", "num_phases", "num_expert_states",
    "num_phase_boundaries", "num_state_boundaries",
    "proprio_finite", "actions_finite", "issues", "passed",
]


def _load_dataset(cfg):
    """Load real data or fall back to synthetic."""
    export_root = getattr(cfg.applied_data, "export_root", "")
    index_json = getattr(cfg.applied_data, "index_json", "")

    if export_root and Path(export_root).exists():
        return load_applied_dataset(export_root, index_json)

    log.warning("Real data not found at %s. Using synthetic fallback.", export_root)
    return generate_synthetic_episodes(num_episodes=10, T=100)


def run_experiment(cfg=None, verbose: bool = False):
    if cfg is None:
        cfg = load_config("experiments/configs/robotics.yaml")
        validate_config(cfg)

    report = start_report(
        EXPERIMENT_ID, EXPERIMENT_NAME,
        "Trust prerequisite: data integrity validation",
        "All episodes pass structural integrity checks",
    )
    capsule = setup_run(cfg, "applied_dataset_audit")
    csv_rows: list[dict] = []
    start = perf_counter()

    dataset = _load_dataset(cfg)
    audit = validate_dataset(dataset)

    log.info("Dataset summary: %s", dataset.summary())
    log.info("Audit result: valid=%d, invalid=%d",
             audit["valid_episodes"], audit["invalid_episodes"])

    lengths = []
    ep_labels = []

    for ep in dataset.episodes:
        issues = validate_episode(ep)
        passed = len(issues) == 0

        row = {
            "episode_id": ep.episode_id,
            "length": ep.length,
            "num_phases": len(np.unique(ep.gt_phase)),
            "num_expert_states": len(set(ep.expert_states)),
            "num_phase_boundaries": len(ep.phase_boundaries),
            "num_state_boundaries": len(ep.expert_state_boundaries),
            "proprio_finite": bool(np.all(np.isfinite(ep.proprio))),
            "actions_finite": bool(np.all(np.isfinite(ep.actions))),
            "issues": "; ".join(issues) if issues else "",
            "passed": passed,
        }
        record_case(report, capsule, csv_rows, ep.episode_id, passed, row,
                     error="; ".join(issues) if issues else None)

        lengths.append(ep.length)
        ep_labels.append(ep.episode_id[-8:])

        if verbose:
            status = "OK" if passed else f"ISSUES: {issues}"
            print(f"  {ep.episode_id}: T={ep.length}, phases={len(np.unique(ep.gt_phase))}, "
                  f"states={len(set(ep.expert_states))}, {status}")

    report.duration_seconds = perf_counter() - start

    # Figure: episode lengths
    if lengths:
        plot_bar(
            ep_labels, lengths,
            "Episode Lengths",
            "Timesteps",
            capsule.figures / "audit_episode_lengths",
            cfg.plotting.formats, cfg.plotting.theme,
            highlight_best=False,
        )

    report.metadata["audit"] = audit
    report.metadata["acceptance_passed"] = audit["all_valid"]

    log.info("AP-00 audit: %d/%d episodes valid",
             audit["valid_episodes"], audit["num_episodes"])

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
