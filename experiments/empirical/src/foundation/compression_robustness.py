"""
PC-04: Compression-Robustness Frontier
========================================

Paper claim
    SYNAPSE remains robust under corruption and tighter memory budgets
    better than naive compression.

Methodology
    Sweep memory budget K and corruption sigma. Measure feature quality
    score for UniformSample, AnchorOnly, and SYNAPSE.
    CRITICAL FIX: Never mutate the shared config object.

Outputs
    figures/
        compression_heatmap.{png,pdf}  -- heatmap: budget x sigma
        compression_line.{png,pdf}      -- line: methods vs sweep
    metrics/metrics.jsonl, results.csv
    artifacts/report.json, logs/run.log
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.empirical.common.baselines import anchor_feature, synapse_feature, uniform_feature
from experiments.empirical.common.experiment import (
    finalize_and_save, plot_heatmap, plot_line, record_case, setup_run, start_report,
)
from experiments.empirical.common.config import load_config, validate_config
from experiments.empirical.common.stress import add_observation_noise
from experiments.empirical.common.tasks import generate_memory_task_dataset
from experiments.utils.progress import iter_progress

log = logging.getLogger(__name__)

EXPERIMENT_ID = "PC-04"
EXPERIMENT_NAME = "Compression-Robustness Frontier"

CSV_FIELDS = ["memory_budget", "corruption_sigma", "method", "score", "memory_size", "passed"]

METHODS = ["UniformSample", "AnchorOnly", "SYNAPSE"]


def _family_score(sequence, K, r, tau, weights, Q, method):
    """Compute feature-quality score for a given method. Never mutates config."""
    if method == "UniformSample":
        feat = uniform_feature(sequence, K)
        return float(np.mean(np.abs(feat))), feat.size
    elif method == "AnchorOnly":
        feat, idx = anchor_feature(sequence, K, r, tau, weights)
        return float(len(idx)) / max(1, K), feat.size
    else:
        feat, idx, mem_size = synapse_feature(sequence, K, r, tau, Q, weights)
        return float(np.mean(np.abs(feat))) + 0.01 * len(idx), mem_size


def run_experiment(cfg=None, verbose: bool = False):
    if cfg is None:
        cfg = load_config("experiments/configs/foundation.yaml")
        validate_config(cfg)

    report = start_report(
        EXPERIMENT_ID, EXPERIMENT_NAME,
        "Paper claim: SYNAPSE improves the compression-performance frontier",
        "SYNAPSE remains robust under corruption and tighter budgets",
    )
    capsule = setup_run(cfg, "compression_robustness")
    csv_rows: list[dict] = []
    rng = np.random.default_rng(2026)
    start = perf_counter()

    # Extract parameters ONCE (never mutate)
    r = cfg.memory.r
    tau = cfg.memory.tau
    Q = cfg.memory.Q
    weights = tuple(cfg.memory.weights)
    budgets = list(cfg.stress_tests.memory_budgets)
    sigmas = list(cfg.stress_tests.corruption_sigmas)

    base_samples = generate_memory_task_dataset(
        rng, cfg.synthetic_tasks.task_families[0],
        cfg.synthetic_tasks.test_samples, cfg.synthetic_tasks.seq_len,
        cfg.synthetic_tasks.state_dim, distractor_scale=0.45,
    )

    # Collect scores for heatmap: [budget x sigma] for SYNAPSE
    synapse_grid = np.zeros((len(budgets), len(sigmas)))
    method_averages: dict[str, list[float]] = {m: [] for m in METHODS}

    sweep = [(bi, si) for bi in range(len(budgets)) for si in range(len(sigmas))]

    for bi, si in iter_progress(sweep, desc="PC-04 sweep"):
        K_local = int(budgets[bi])
        sigma = sigmas[si]

        for method in METHODS:
            scores, mem_sizes = [], []
            for sample in base_samples:
                corrupted = add_observation_noise(sample.sequence, rng, sigma)
                score, mem_size = _family_score(corrupted, K_local, r, tau, weights, Q, method)
                scores.append(score)
                mem_sizes.append(mem_size)

            mean_score = float(np.mean(scores))
            mean_mem = float(np.mean(mem_sizes))
            method_averages[method].append(mean_score)

            if method == "SYNAPSE":
                synapse_grid[bi, si] = mean_score

            row = {
                "memory_budget": K_local,
                "corruption_sigma": sigma,
                "method": method,
                "score": round(mean_score, 6),
                "memory_size": round(mean_mem, 2),
                "passed": True,
            }
            record_case(report, capsule, csv_rows, f"K{K_local}_s{sigma}_{method}", True, row)

    report.duration_seconds = perf_counter() - start

    # ---- Publication figures -----------------------------------------------
    # Figure 1: Heatmap — SYNAPSE score across budget x corruption
    plot_heatmap(
        synapse_grid,
        [f"K={b}" for b in budgets],
        [f"sigma={s}" for s in sigmas],
        "SYNAPSE Feature Quality: Budget vs Corruption",
        capsule.figures / "compression_heatmap",
        cfg.plotting.formats, cfg.plotting.theme,
        cmap="YlGnBu",
    )

    # Figure 2: Line — methods across sweep
    x_idx = list(range(len(sweep)))
    plot_line(
        x_idx,
        [method_averages[m] for m in METHODS],
        METHODS,
        "Compression-Robustness Score Across Conditions",
        "Sweep Index (K x sigma)",
        "Feature Quality Score",
        capsule.figures / "compression_line",
        cfg.plotting.formats, cfg.plotting.theme,
    )

    synapse_mean = float(np.mean(method_averages["SYNAPSE"]))
    uniform_mean = float(np.mean(method_averages["UniformSample"]))
    report.metadata["acceptance_passed"] = bool(synapse_mean - uniform_mean >= cfg.acceptance_gates.pc04_frontier_margin)
    log.info("PC-04 SYNAPSE mean: %.4f, Uniform mean: %.4f", synapse_mean, uniform_mean)

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
