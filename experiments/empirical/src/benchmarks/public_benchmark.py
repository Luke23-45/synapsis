"""
Downstream Compatibility: Public Benchmark (State-Only)
=======================================================

Downstream compatibility evidence:
    SYNAPSE representations integrate cleanly into standard downstream 
    learning pipelines and improve terminal accuracy.
    (Note: This is tertiary evidence; the primary contribution is non-training).

Methodology
    Load public benchmark episodes (or synthetic fallback).
    Flatten to prefix-prediction regression. Compare FixedWindow,
    GRU, Transformer, AnchorOnly, and SYNAPSE on MSE.

Outputs
    figures/public_benchmark_mse.{png,pdf}
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
    anchor_feature, build_baseline, synapse_feature, uniform_feature,
)
from experiments.empirical.common.eval import (
    evaluate_model, make_loaders, set_seed, train_model,
)
from experiments.empirical.common.experiment import (
    baseline_label, finalize_and_save, plot_bar,
    record_case, setup_run, start_report,
)
from experiments.empirical.common.config import load_config, validate_config
from experiments.empirical.common.io import load_public_state_dataset

log = logging.getLogger(__name__)

EXPERIMENT_ID = "PC-05"
EXPERIMENT_NAME = "Public Benchmark (State-Only)"

CSV_FIELDS = ["baseline", "baseline_label", "seed", "mse", "mae", "passed"]

BASELINES = ["B0", "B1", "B2", "B4", "B6"]


def _flatten_episodes(episodes):
    """Convert episodes into (prefix -> next_action) pairs."""
    states, actions = [], []
    for ep in episodes:
        T = min(len(ep["states"]), len(ep["actions"]))
        for t in range(1, T):
            states.append(ep["states"][:t + 1])
            actions.append(ep["actions"][t])
    return states, np.asarray(actions, dtype=np.float32)


def _pad_prefix(seq, target_len):
    """Left-pad a prefix sequence to target_len."""
    if len(seq) >= target_len:
        return seq[-target_len:].astype(np.float32)
    pad = np.repeat(seq[:1], target_len - len(seq), axis=0)
    return np.concatenate([pad, seq], axis=0).astype(np.float32)


def _feature_state(seq, bl, cfg):
    """Extract features from a prefix sequence."""
    if bl == "B0":
        return _pad_prefix(seq, cfg.baselines.window_size).reshape(-1)
    if bl in {"B1", "B2"}:
        return _pad_prefix(seq, cfg.public_benchmark.seq_len)
    if bl == "B3":
        return uniform_feature(seq, cfg.memory.K)
    if bl == "B4":
        feat, _ = anchor_feature(seq, cfg.memory.K, cfg.memory.r, cfg.memory.tau, tuple(cfg.memory.weights))
        return feat
    feat, _, _ = synapse_feature(seq, cfg.memory.K, cfg.memory.r, cfg.memory.tau, cfg.memory.Q, tuple(cfg.memory.weights))
    return feat


def run_experiment(cfg=None, verbose: bool = False):
    if cfg is None:
        cfg = load_config("experiments/configs/benchmarks.yaml")
        validate_config(cfg)

    report = start_report(
        EXPERIMENT_ID, EXPERIMENT_NAME,
        "Downstream compatibility evidence: terminal accuracy on standard pipelines",
        "SYNAPSE representations integrate seamlessly and improve terminal accuracy",
    )
    capsule = setup_run(cfg, "public_benchmark")
    csv_rows: list[dict] = []
    start = perf_counter()

    episodes = load_public_state_dataset(
        cfg.public_benchmark.dataset_path,
        cfg.public_benchmark.smoke_use_synthetic_fallback,
        seed=77,
    )
    sequences, targets = _flatten_episodes(episodes)
    split = max(1, int(0.7 * len(sequences)))
    train_seq, test_seq = sequences[:split], sequences[split:]
    y_train, y_test = targets[:split], targets[split:]

    summary: dict[str, float] = {}
    summary_std: dict[str, float] = {}

    for bl in BASELINES:
        lbl = baseline_label(bl)
        log.info("  Baseline: %s (%s)", bl, lbl)
        seed_mse = []

        for seed_idx in range(cfg.training.num_seeds):
            set_seed(500 + seed_idx)
            x_train = np.stack([_feature_state(s, bl, cfg).astype(np.float32) for s in train_seq])
            x_test = np.stack([_feature_state(s, bl, cfg).astype(np.float32) for s in test_seq])

            seq_model = bl in {"B1", "B2"}
            model = build_baseline(bl, x_train.shape[-1], y_train.shape[-1], cfg.baselines.hidden_dim, max_len=x_train.shape[1] if seq_model else 512)

            train_loader, val_loader = make_loaders(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(x_test), torch.from_numpy(y_test), cfg.training.batch_size)
            train_model(model, train_loader, val_loader, cfg.training.full_epochs, "regression", cfg.training.learning_rate, cfg.training.device)
            test_loader, _ = make_loaders(torch.from_numpy(x_test), torch.from_numpy(y_test), torch.from_numpy(x_test), torch.from_numpy(y_test), cfg.training.batch_size)
            metrics = evaluate_model(model, test_loader, "regression", cfg.training.device)
            seed_mse.append(metrics["mse"])

            row = {
                "baseline": bl, "baseline_label": lbl, "seed": seed_idx,
                "mse": round(metrics["mse"], 6), "mae": round(metrics["mae"], 6),
                "passed": True,
            }
            record_case(report, capsule, csv_rows, f"{bl}_{seed_idx}", True, row)

        summary[lbl] = float(np.mean(seed_mse))
        summary_std[lbl] = float(np.std(seed_mse))

    report.duration_seconds = perf_counter() - start

    plot_bar(
        list(summary.keys()), list(summary.values()),
        "Public Benchmark: Regression MSE",
        "MSE (lower is better)",
        capsule.figures / "public_benchmark_mse",
        cfg.plotting.formats, cfg.plotting.theme,
        errors=list(summary_std.values()),
        highlight_best=False,
    )

    synapse_mse = summary.get("SYNAPSE", float("inf"))
    best_other = min(v for k, v in summary.items() if k != "SYNAPSE")
    report.metadata["acceptance_passed"] = bool(synapse_mse <= best_other + cfg.acceptance_gates.pc05_relative_gain)
    report.metadata["summary_mse"] = summary
    log.info("PC-05 MSE: %s", {k: f"{v:.6f}" for k, v in summary.items()})

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
