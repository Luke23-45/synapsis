"""
PC-06: Ablation Study
======================

Paper claim
    Gains arise from the intended memory mechanism; ablating topology
    or anchor scoring degrades performance measurably.

Methodology
    Run downstream accuracy (not just feature norm) across ablation
    variants on the topology dataset. Compare:
      - Full SYNAPSE
      - Remove topology (anchors only)
      - Raw delta scorer (r=1, no refractory)
      - Proxy-only topology (no persistence diagrams)

Outputs
    figures/ablation_accuracy.{png,pdf}
    metrics/metrics.jsonl, results.csv
    artifacts/report.json, logs/run.log
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.empirical.common.baselines import (
    anchor_feature, build_baseline, proxy_topology_features, synapse_feature,
)
from experiments.empirical.common.eval import (
    aggregate_seed_metrics, evaluate_model, make_loaders, set_seed, train_model,
)
from experiments.empirical.common.experiment import (
    finalize_and_save, plot_bar, record_case, setup_run, start_report,
)
from experiments.empirical.common.config import load_config, validate_config
from experiments.empirical.common.tasks import generate_topology_dataset

log = logging.getLogger(__name__)

EXPERIMENT_ID = "PC-06"
EXPERIMENT_NAME = "Ablation Study"

CSV_FIELDS = ["ablation", "seed", "accuracy", "confidence", "passed"]

ABLATIONS = ["Full SYNAPSE", "Remove Topology", "Raw Delta Scorer", "Proxy Only"]


def _ablation_feature(seq, ablation, cfg):
    """Extract features according to ablation variant."""
    K = cfg.memory.K
    r = cfg.memory.r
    tau = cfg.memory.tau
    weights = tuple(cfg.memory.weights)

    if ablation == "Remove Topology":
        feat, _ = anchor_feature(seq, K, r, tau, weights)
        return feat
    if ablation == "Raw Delta Scorer":
        feat, _ = anchor_feature(seq, K, 1, tau, weights)
        return feat
    if ablation == "Proxy Only":
        feat, _ = anchor_feature(seq, K, r, tau, weights)
        return np.concatenate([feat, proxy_topology_features(seq, K)])
    # Full SYNAPSE
    feat, _, _ = synapse_feature(seq, K, r, tau, cfg.memory.Q, weights)
    return feat


def run_experiment(cfg=None, verbose: bool = False):
    if cfg is None:
        cfg = load_config("experiments/configs/benchmarks.yaml")
        validate_config(cfg)

    report = start_report(
        EXPERIMENT_ID, EXPERIMENT_NAME,
        "Paper claim: gains arise from the intended memory mechanism",
        "Ablations isolate topology and anchor scoring contributions",
    )
    capsule = setup_run(cfg, "ablation_study")
    csv_rows: list[dict] = []
    rng = np.random.default_rng(2026)
    results: dict[str, dict] = {}
    start = perf_counter()

    train = generate_topology_dataset(rng, cfg.synthetic_tasks.topology_train_samples, cfg.synthetic_tasks.seq_len, cfg.synthetic_tasks.state_dim, noise_std=0.03)
    test = generate_topology_dataset(rng, cfg.synthetic_tasks.topology_test_samples, cfg.synthetic_tasks.seq_len, cfg.synthetic_tasks.state_dim, noise_std=0.08)

    for ablation in ABLATIONS:
        log.info("  Ablation: %s", ablation)
        per_seed = []

        for seed_idx in range(cfg.training.num_seeds):
            set_seed(seed_idx + 200)
            x_train = np.stack([_ablation_feature(s.sequence, ablation, cfg).astype(np.float32) for s in train])
            y_train = np.asarray([s.label for s in train], dtype=np.int64)
            x_test = np.stack([_ablation_feature(s.sequence, ablation, cfg).astype(np.float32) for s in test])
            y_test = np.asarray([s.label for s in test], dtype=np.int64)

            model = build_baseline("B6", x_train.shape[1], int(y_train.max()) + 1, cfg.baselines.hidden_dim)
            train_loader, val_loader = make_loaders(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(x_test), torch.from_numpy(y_test), cfg.training.batch_size)
            train_model(model, train_loader, val_loader, cfg.training.full_epochs, "classification", cfg.training.learning_rate, cfg.training.device)
            test_loader, _ = make_loaders(torch.from_numpy(x_test), torch.from_numpy(y_test), torch.from_numpy(x_test), torch.from_numpy(y_test), cfg.training.batch_size)
            metrics = evaluate_model(model, test_loader, "classification", cfg.training.device)

            row = {"ablation": ablation, "seed": seed_idx, "accuracy": round(metrics["accuracy"], 6), "confidence": round(metrics["confidence"], 6), "passed": True}
            record_case(report, capsule, csv_rows, f"{ablation}_{seed_idx}", True, row)
            per_seed.append(metrics)

        results[ablation] = aggregate_seed_metrics(per_seed, "accuracy")

    report.duration_seconds = perf_counter() - start

    plot_bar(
        list(results.keys()),
        [results[a]["mean"] for a in results],
        "Ablation Study: Classification Accuracy",
        "Accuracy",
        capsule.figures / "ablation_accuracy",
        cfg.plotting.formats, cfg.plotting.theme,
        errors=[results[a]["std"] for a in results],
    )

    full_acc = results["Full SYNAPSE"]["mean"]
    no_topo_acc = results["Remove Topology"]["mean"]
    drop = full_acc - no_topo_acc
    report.metadata["acceptance_passed"] = bool(drop >= cfg.acceptance_gates.pc06_min_drop)
    report.metadata["ablation_accuracy"] = {a: round(results[a]["mean"], 6) for a in results}
    log.info("PC-06 topology drop: %.4f (gate: %.4f)", drop, cfg.acceptance_gates.pc06_min_drop)

    return finalize_and_save(report, capsule, csv_rows, CSV_FIELDS, cfg)


if __name__ == "__main__":
    config_path = "experiments/configs/benchmarks.yaml"
    for arg in sys.argv:
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
    cfg = load_config(config_path)
    validate_config(cfg)
    v = "--verbose" in sys.argv or "-v" in sys.argv
    report = run_experiment(cfg=cfg, verbose=v)
    print(f"[{report.status}] {EXPERIMENT_ID}: {EXPERIMENT_NAME}")
    sys.exit(0 if report.status == "PASS" else 1)
