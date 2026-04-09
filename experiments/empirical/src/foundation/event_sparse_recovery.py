"""
Controlled Mechanistic Study: Event-Sparse Recovery
===================================================

Controlled support evidence:
    SYNAPSE anchor selection recovers salient events more robustly
    than naive schemes under nuisance variation.

Methodology
    Generate synthetic trajectories with known event positions.
    Compare four methods (Oracle, DeltaThreshold, UniformSample, SYNAPSE)
    on precision, recall, F1, and localization error across noise and
    distractor sweeps.

Outputs
    figures/
        event_recovery_f1.{png,pdf}          -- grouped bar (methods x noise)
        event_recovery_localization.{png,pdf} -- localization error comparison
    metrics/metrics.jsonl, results.csv
    artifacts/report.json
    logs/run.log
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.empirical.common.baselines import anchor_feature
from experiments.empirical.common.experiment import (
    baseline_label, finalize_and_save, plot_bar, plot_grouped_bar,
    record_case, setup_run, start_report,
)
from experiments.empirical.common.config import load_config, validate_config
from experiments.empirical.common.tasks import generate_event_sparse_sequence
from experiments.utils.progress import iter_progress

log = logging.getLogger(__name__)

EXPERIMENT_ID = "PC-01"
EXPERIMENT_NAME = "Event-Sparse Recovery"
FORMAL_REF = "Controlled support evidence: salient-event recovery under nuisance variation"
CLAIM = "SYNAPSE anchor selection recovers meaningful events more robustly than naive schemes"

CSV_FIELDS = [
    "noise_std", "distractor_scale", "trial", "method",
    "precision", "recall", "f1", "localization_error",
    "budget_efficiency", "passed",
]

METHODS = ["Oracle", "DeltaThreshold", "UniformSample", "SYNAPSE"]


def _match_metrics(
    pred: List[int],
    true: List[int],
    tolerance: int = 2,
) -> Tuple[float, float, float, float]:
    """Compute precision, recall, F1, and mean localization error."""
    true_remaining = set(true)
    tp = 0
    localization: List[float] = []

    for p in pred:
        match = None
        for t in sorted(true_remaining):
            if abs(p - t) <= tolerance:
                match = t
                break
        if match is not None:
            true_remaining.remove(match)
            tp += 1
            localization.append(abs(p - match))

    fp = max(0, len(pred) - tp)
    fn = len(true_remaining)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    loc = float(np.mean(localization)) if localization else float(len(pred) + len(true))
    return precision, recall, f1, loc


def run_experiment(cfg=None, verbose: bool = False):
    if cfg is None:
        cfg = load_config("experiments/configs/foundation.yaml")
        validate_config(cfg)

    report = start_report(EXPERIMENT_ID, EXPERIMENT_NAME, FORMAL_REF, CLAIM)
    capsule = setup_run(cfg, "event_sparse_recovery")
    csv_rows: list[dict] = []

    rng = np.random.default_rng(42)
    K = cfg.memory.K
    r = cfg.memory.r
    tau = cfg.memory.tau
    weights = tuple(cfg.memory.weights)

    # Aggregators for publication figure
    method_f1_by_noise: dict[str, dict[float, list[float]]] = {m: {} for m in METHODS}

    start = perf_counter()

    sweep = [
        (noise, distractor)
        for noise in cfg.synthetic_tasks.noise_levels
        for distractor in cfg.synthetic_tasks.distractor_levels
    ]

    for noise_std, distractor in iter_progress(sweep, desc="PC-01 sweep"):
        for trial in range(cfg.synthetic_tasks.event_recovery_trials):
            sequence, truth = generate_event_sparse_sequence(
                rng=rng,
                T=cfg.synthetic_tasks.seq_len,
                d=cfg.synthetic_tasks.state_dim,
                num_events=int(rng.integers(3, 6)),
                noise_std=noise_std,
                distractor_scale=distractor,
                smooth=bool(trial % 2),
            )

            # Compute deltas
            diffs = np.zeros(len(sequence), dtype=np.float32)
            diffs[1:] = np.linalg.norm(np.diff(sequence, axis=0), axis=1)

            delta_pred = np.where(diffs > tau)[0].tolist()[:K]
            uniform_pred = np.linspace(
                0, len(sequence) - 1, num=min(K, len(sequence)), dtype=int,
            ).tolist()
            _, synapse_pred = anchor_feature(sequence, K, r, tau, weights)

            pred_map = {
                "Oracle": truth,
                "DeltaThreshold": delta_pred,
                "UniformSample": uniform_pred,
                "SYNAPSE": synapse_pred,
            }

            for method, pred in pred_map.items():
                precision, recall, f1, loc = _match_metrics(pred, truth)
                budget_eff = len(pred) / max(1, K)

                row = {
                    "noise_std": noise_std,
                    "distractor_scale": distractor,
                    "trial": trial,
                    "method": method,
                    "precision": round(precision, 6),
                    "recall": round(recall, 6),
                    "f1": round(f1, 6),
                    "localization_error": round(loc, 4),
                    "budget_efficiency": round(budget_eff, 4),
                    "passed": True,
                }
                record_case(
                    report, capsule, csv_rows,
                    f"{method}_n{noise_std}_d{distractor}_t{trial}",
                    True, row,
                )
                method_f1_by_noise[method].setdefault(noise_std, []).append(f1)

    report.duration_seconds = perf_counter() - start

    # ---- Publication figures -----------------------------------------------
    # Figure 1: Grouped bar — mean F1 per method per noise level
    noise_levels = sorted(cfg.synthetic_tasks.noise_levels)
    noise_labels = [f"sigma={n}" for n in noise_levels]
    grouped_vals: dict[str, list[float]] = {}
    grouped_errs: dict[str, list[float]] = {}
    for method in METHODS:
        grouped_vals[method] = []
        grouped_errs[method] = []
        for noise in noise_levels:
            vals = method_f1_by_noise[method].get(noise, [0.0])
            grouped_vals[method].append(float(np.mean(vals)))
            grouped_errs[method].append(float(np.std(vals)))

    plot_grouped_bar(
        noise_labels, METHODS, grouped_vals,
        "Event Recovery F1 Score by Method and Noise Level",
        "F1 Score",
        capsule.figures / "event_recovery_f1",
        cfg.plotting.formats, cfg.plotting.theme,
        errors=grouped_errs,
    )

    # Figure 2: Summary bar — overall mean F1 per method
    overall_f1 = {m: float(np.mean([v for vs in method_f1_by_noise[m].values() for v in vs])) for m in METHODS}
    overall_std = {m: float(np.std([v for vs in method_f1_by_noise[m].values() for v in vs])) for m in METHODS}

    plot_bar(
        list(overall_f1.keys()),
        list(overall_f1.values()),
        "Overall Mean Event Recovery F1",
        "F1 Score",
        capsule.figures / "event_recovery_overall",
        cfg.plotting.formats, cfg.plotting.theme,
        errors=list(overall_std.values()),
    )

    # Acceptance gate
    margin = overall_f1["SYNAPSE"] - max(overall_f1["DeltaThreshold"], overall_f1["UniformSample"])
    report.metadata["summary_f1"] = overall_f1
    report.metadata["acceptance_margin"] = round(margin, 6)
    report.metadata["acceptance_passed"] = bool(margin >= cfg.acceptance_gates.pc01_f1_margin)

    log.info("PC-01 F1 summary: %s", {k: f"{v:.4f}" for k, v in overall_f1.items()})
    log.info("PC-01 acceptance margin: %.4f (gate: %.4f)", margin, cfg.acceptance_gates.pc01_f1_margin)

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
