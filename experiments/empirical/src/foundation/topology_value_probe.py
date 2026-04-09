"""
PC-03: Topology Value Probe
============================

Paper claim
    Topology-enhanced memory features resolve structure that local
    event anchors alone miss.

Methodology
    Train classifiers on topology-variant features.
    Compare anchors-only vs anchors+proxy vs full SYNAPSE.

Outputs
    figures/topology_probe_accuracy.{png,pdf}
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

EXPERIMENT_ID = "PC-03"
EXPERIMENT_NAME = "Topology Value Probe"

CSV_FIELDS = ["variant", "seed", "accuracy", "confidence", "passed"]

VARIANTS = ["AnchorOnly", "Anchor+Proxy", "FullSYNAPSE"]


def _dataset_features(samples, variant, cfg):
    x, y = [], []
    for sample in samples:
        seq = sample.sequence
        anchor_feat, _ = anchor_feature(seq, cfg.memory.K, cfg.memory.r, cfg.memory.tau, tuple(cfg.memory.weights))
        if variant == "AnchorOnly":
            feat = anchor_feat
        elif variant == "Anchor+Proxy":
            feat = np.concatenate([anchor_feat, proxy_topology_features(seq, cfg.memory.K)])
        else:
            feat, _, _ = synapse_feature(seq, cfg.memory.K, cfg.memory.r, cfg.memory.tau, cfg.memory.Q, tuple(cfg.memory.weights))
        x.append(feat.astype(np.float32))
        y.append(sample.label)
    return np.stack(x), np.asarray(y, dtype=np.int64)


def run_experiment(cfg=None, verbose: bool = False):
    if cfg is None:
        cfg = load_config("experiments/configs/foundation.yaml")
        validate_config(cfg)

    report = start_report(
        EXPERIMENT_ID, EXPERIMENT_NAME,
        "Paper claim: topology contributes beyond anchors alone",
        "Topology-enhanced features resolve structure that anchors miss",
    )
    capsule = setup_run(cfg, "topology_value_probe")
    csv_rows: list[dict] = []
    rng = np.random.default_rng(1234)
    results: dict[str, dict] = {}
    start = perf_counter()

    train = generate_topology_dataset(rng, cfg.synthetic_tasks.topology_train_samples, cfg.synthetic_tasks.seq_len, cfg.synthetic_tasks.state_dim, noise_std=0.03)
    test = generate_topology_dataset(rng, cfg.synthetic_tasks.topology_test_samples, cfg.synthetic_tasks.seq_len, cfg.synthetic_tasks.state_dim, noise_std=0.08)

    for variant in VARIANTS:
        log.info("  Variant: %s", variant)
        per_seed = []
        for seed_idx in range(cfg.training.num_seeds):
            set_seed(seed_idx + 99)
            x_train, y_train = _dataset_features(train, variant, cfg)
            x_test, y_test = _dataset_features(test, variant, cfg)

            model = build_baseline("B5" if variant != "FullSYNAPSE" else "B6", x_train.shape[1], int(y_train.max()) + 1, cfg.baselines.hidden_dim)
            train_loader, val_loader = make_loaders(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(x_test), torch.from_numpy(y_test), cfg.training.batch_size)
            train_model(model, train_loader, val_loader, cfg.training.full_epochs, "classification", cfg.training.learning_rate, cfg.training.device)
            test_loader, _ = make_loaders(torch.from_numpy(x_test), torch.from_numpy(y_test), torch.from_numpy(x_test), torch.from_numpy(y_test), cfg.training.batch_size)
            metrics = evaluate_model(model, test_loader, "classification", cfg.training.device)

            row = {"variant": variant, "seed": seed_idx, "accuracy": round(metrics["accuracy"], 6), "confidence": round(metrics["confidence"], 6), "passed": True}
            record_case(report, capsule, csv_rows, f"{variant}_{seed_idx}", True, row)
            per_seed.append(metrics)
        results[variant] = aggregate_seed_metrics(per_seed, "accuracy")

    report.duration_seconds = perf_counter() - start

    plot_bar(
        list(results.keys()),
        [results[v]["mean"] for v in results],
        "Topology Probe: Classification Accuracy",
        "Accuracy",
        capsule.figures / "topology_probe_accuracy",
        cfg.plotting.formats, cfg.plotting.theme,
        errors=[results[v]["std"] for v in results],
    )

    gain = results["FullSYNAPSE"]["mean"] - results["AnchorOnly"]["mean"]
    report.metadata["acceptance_passed"] = bool(gain >= cfg.acceptance_gates.pc03_topology_gain)
    report.metadata["variant_accuracy"] = {v: round(results[v]["mean"], 6) for v in results}
    log.info("PC-03 topology gain: %.4f (gate: %.4f)", gain, cfg.acceptance_gates.pc03_topology_gain)

    return finalize_and_save(report, capsule, csv_rows, CSV_FIELDS, cfg)


if __name__ == "__main__":
    config_path = "experiments/configs/foundation.yaml"
    for arg in sys.argv:
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
    cfg = load_config(config_path)
    validate_config(cfg)
    v = "--verbose" in sys.argv or "-v" in sys.argv
    report = run_experiment(cfg=cfg, verbose=v)
    print(f"[{report.status}] {EXPERIMENT_ID}: {EXPERIMENT_NAME}")
    sys.exit(0 if report.status == "PASS" else 1)
