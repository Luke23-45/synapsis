"""
PC-02: Memory Sufficiency Under Distractors
=============================================

Paper claim
    SYNAPSE's compressed memory preserves task-relevant information
    better than fixed-window or naive compression baselines.

Methodology
    Train lightweight classifiers on features from each baseline
    across 4 task families. Report accuracy +/- std over seeds.

Outputs
    figures/
        memory_sufficiency_grouped.{png,pdf}   -- accuracy by family x method
        memory_sufficiency_overall.{png,pdf}   -- overall accuracy comparison
    metrics/metrics.jsonl, results.csv
    artifacts/report.json
    logs/run.log
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
    anchor_feature, build_baseline, proxy_topology_features,
    synapse_feature, uniform_feature,
)
from experiments.empirical.common.eval import (
    aggregate_seed_metrics, evaluate_model, make_loaders, set_seed, train_model,
)
from experiments.empirical.common.experiment import (
    baseline_label, finalize_and_save, plot_grouped_bar, plot_bar,
    record_case, setup_run, start_report,
)
from experiments.empirical.common.config import load_config, validate_config
from experiments.empirical.common.tasks import generate_memory_task_dataset
from experiments.utils.progress import iter_progress

log = logging.getLogger(__name__)

EXPERIMENT_ID = "PC-02"
EXPERIMENT_NAME = "Memory Sufficiency Under Distractors"

CSV_FIELDS = [
    "family", "baseline", "baseline_label", "seed",
    "accuracy", "confidence", "memory_size", "train_time", "passed",
]


def _feature_dataset(samples, baseline_name, cfg):
    """Extract features from samples using specified baseline method."""
    feats, labels, mem_sizes = [], [], []

    for sample in samples:
        seq = sample.sequence
        if baseline_name == "B0":
            feat = seq[-cfg.baselines.window_size:].reshape(-1)
        elif baseline_name == "B3":
            feat = uniform_feature(seq, cfg.memory.K)
        elif baseline_name == "B4":
            feat, _ = anchor_feature(seq, cfg.memory.K, cfg.memory.r, cfg.memory.tau, tuple(cfg.memory.weights))
        elif baseline_name == "B5":
            anchor_feat, _ = anchor_feature(seq, cfg.memory.K, cfg.memory.r, cfg.memory.tau, tuple(cfg.memory.weights))
            feat = np.concatenate([anchor_feat, proxy_topology_features(seq, cfg.memory.K)])
        elif baseline_name == "B6":
            feat, _, mem_size = synapse_feature(seq, cfg.memory.K, cfg.memory.r, cfg.memory.tau, cfg.memory.Q, tuple(cfg.memory.weights))
            mem_sizes.append(mem_size)
        else:
            feat = seq.reshape(-1) if seq.ndim > 1 else seq

        feats.append(feat.astype(np.float32))
        labels.append(sample.label)
        if baseline_name != "B6":
            mem_sizes.append(int(np.asarray(feat).size))

    return np.stack(feats), np.asarray(labels, dtype=np.int64), float(np.mean(mem_sizes))


def _run_one_seed(train_samples, val_samples, test_samples, baseline_name, cfg, seed):
    """Train and evaluate one seed of one baseline."""
    set_seed(seed)
    x_train, y_train, mem_size = _feature_dataset(train_samples, baseline_name, cfg)
    x_val, y_val, _ = _feature_dataset(val_samples, baseline_name, cfg)
    x_test, y_test, _ = _feature_dataset(test_samples, baseline_name, cfg)

    seq_model = baseline_name in {"B1", "B2"}

    model = build_baseline(
        baseline_name, x_train.shape[-1],
        int(y_train.max()) + 1, cfg.baselines.hidden_dim,
        max_len=x_train.shape[1] if seq_model else 512,
    )

    train_loader, val_loader = make_loaders(
        torch.from_numpy(x_train), torch.from_numpy(y_train),
        torch.from_numpy(x_val), torch.from_numpy(y_val),
        cfg.training.batch_size,
    )
    fit = train_model(
        model, train_loader, val_loader,
        cfg.training.full_epochs, "classification",
        cfg.training.learning_rate, cfg.training.device,
    )

    test_loader, _ = make_loaders(
        torch.from_numpy(x_test), torch.from_numpy(y_test),
        torch.from_numpy(x_test), torch.from_numpy(y_test),
        cfg.training.batch_size,
    )
    metrics = evaluate_model(model, test_loader, "classification", cfg.training.device)
    metrics["memory_size"] = mem_size
    metrics["train_time"] = fit.train_time
    return metrics


def run_experiment(cfg=None, verbose: bool = False):
    if cfg is None:
        cfg = load_config("experiments/configs/foundation.yaml")
        validate_config(cfg)

    report = start_report(
        EXPERIMENT_ID, EXPERIMENT_NAME,
        "Paper claim: compressed memory preserves task-relevant information",
        "SYNAPSE memory preserves task accuracy better than baselines",
    )
    capsule = setup_run(cfg, "memory_sufficiency")
    csv_rows: list[dict] = []
    rng = np.random.default_rng(cfg.training.num_seeds)
    start = perf_counter()

    families = list(cfg.synthetic_tasks.task_families)
    baselines = list(cfg.baselines.families)
    baseline_scores: dict[tuple, dict] = {}

    total_steps = len(families) * len(baselines) * cfg.training.num_seeds
    step = 0

    for family in families:
        log.info("  Task family: %s", family)
        train_samples = generate_memory_task_dataset(rng, family, cfg.synthetic_tasks.train_samples, cfg.synthetic_tasks.seq_len, cfg.synthetic_tasks.state_dim, 0.25)
        val_samples = generate_memory_task_dataset(rng, family, cfg.synthetic_tasks.val_samples, cfg.synthetic_tasks.seq_len, cfg.synthetic_tasks.state_dim, 0.35)
        test_samples = generate_memory_task_dataset(rng, family, cfg.synthetic_tasks.test_samples, cfg.synthetic_tasks.seq_len, cfg.synthetic_tasks.state_dim, 0.55)

        for bl in baselines:
            per_seed = []
            for seed_idx in range(cfg.training.num_seeds):
                step += 1
                if verbose:
                    print(f"  [{step}/{total_steps}] {family} / {baseline_label(bl)} / seed {seed_idx}")

                metrics = _run_one_seed(train_samples, val_samples, test_samples, bl, cfg, seed=100 * cfg.training.num_seeds + seed_idx)
                row = {
                    "family": family,
                    "baseline": bl,
                    "baseline_label": baseline_label(bl),
                    "seed": seed_idx,
                    "accuracy": round(metrics["accuracy"], 6),
                    "confidence": round(metrics["confidence"], 6),
                    "memory_size": round(metrics["memory_size"], 2),
                    "train_time": round(metrics["train_time"], 3),
                    "passed": True,
                }
                record_case(report, capsule, csv_rows, f"{family}_{bl}_{seed_idx}", True, row)
                per_seed.append(metrics)

            baseline_scores[(family, bl)] = aggregate_seed_metrics(per_seed, "accuracy")

    report.duration_seconds = perf_counter() - start

    # ---- Publication figures -----------------------------------------------
    # Figure 1: Grouped bar — accuracy by family for each method
    method_labels = [baseline_label(bl) for bl in baselines]
    grouped_vals: dict[str, list[float]] = {baseline_label(bl): [] for bl in baselines}
    grouped_errs: dict[str, list[float]] = {baseline_label(bl): [] for bl in baselines}
    for bl in baselines:
        lbl = baseline_label(bl)
        for family in families:
            stats = baseline_scores[(family, bl)]
            grouped_vals[lbl].append(stats["mean"])
            grouped_errs[lbl].append(stats["std"])

    family_labels = [f.replace("_", " ").title() for f in families]
    plot_grouped_bar(
        family_labels, method_labels, grouped_vals,
        "Memory Sufficiency: Accuracy by Task Family",
        "Test Accuracy",
        capsule.figures / "memory_sufficiency_grouped",
        cfg.plotting.formats, cfg.plotting.theme,
        errors=grouped_errs,
    )

    # Figure 2: Overall mean accuracy per baseline
    overall_mean = {baseline_label(bl): float(np.mean([baseline_scores[(f, bl)]["mean"] for f in families])) for bl in baselines}
    overall_std = {baseline_label(bl): float(np.mean([baseline_scores[(f, bl)]["std"] for f in families])) for bl in baselines}

    plot_bar(
        list(overall_mean.keys()), list(overall_mean.values()),
        "Overall Memory Sufficiency",
        "Mean Accuracy",
        capsule.figures / "memory_sufficiency_overall",
        cfg.plotting.formats, cfg.plotting.theme,
        errors=list(overall_std.values()),
    )

    # Acceptance
    synapse_acc = overall_mean.get("SYNAPSE", 0.0)
    uniform_acc = overall_mean.get("UniformSample", 0.0)
    report.metadata["acceptance_passed"] = bool(synapse_acc - uniform_acc >= cfg.acceptance_gates.pc02_accuracy_margin)
    report.metadata["overall_accuracy"] = overall_mean

    log.info("PC-02 overall: %s", {k: f"{v:.4f}" for k, v in overall_mean.items()})

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
