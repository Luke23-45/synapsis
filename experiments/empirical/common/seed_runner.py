"""
experiments/empirical/common/seed_runner.py

Runs an experiment function across multiple seeds and aggregates results.

Phase 2 Shared Infrastructure — see docs/implementation/phase2_empirical_validation/07_shared_infrastructure.md §4
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

from experiments.utils.model_io import create_run_capsule, save_config_snapshot
from .metrics import aggregate_seeds, paired_ttest, bonferroni_correct

log = logging.getLogger(__name__)


def _set_all_seeds(seed: int) -> None:
    """Set seeds for all RNG backends: Python, NumPy, and PyTorch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    # If pytorch_lightning is available, use seed_everything for full coverage
    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed, workers=True)
    except ImportError:
        pass


def run_multi_seed(
    experiment_fn: Callable[[Any, int], Dict[str, float]],
    config: Any,
    seeds: List[int],
    experiment_id: str,
    output_dir: str = "experiments/outputs/empirical",
) -> Dict[str, Any]:
    """
    Run experiment_fn(config, seed) for each seed.
    Collect results, compute statistics, save to JSON.

    Parameters
    ----------
    experiment_fn : callable
        Signature: (config, seed) -> Dict[str, float]
        Must return a flat dict of metric_name -> value.
    config : Any
        Experiment configuration (passed through to experiment_fn).
    seeds : List[int]
        Random seeds to run.
    experiment_id : str
        e.g. "EMP-01"
    output_dir : str
        Where to save results JSON.

    Returns
    -------
    Dict with per-seed results, aggregated statistics, and metadata.
    """
    per_seed_results: List[Dict[str, float]] = []
    per_seed_times: List[float] = []

    capsule = create_run_capsule(output_dir, experiment_id)
    save_config_snapshot(capsule, config)

    raw_dir = capsule.artifacts / "data" / "trials"
    raw_dir.mkdir(parents=True, exist_ok=True)

    original_output_dir = getattr(config, "output_dir", None)
    config.output_dir = str(raw_dir)

    for i, seed in enumerate(seeds):
        log.info("[%s] Running seed %d (%d/%d)", experiment_id, seed, i + 1, len(seeds))
        _set_all_seeds(seed)

        t0 = time.perf_counter()
        result = experiment_fn(config, seed)
        elapsed = time.perf_counter() - t0

        per_seed_results.append(result)
        per_seed_times.append(elapsed)
        log.info("[%s] Seed %d completed in %.2fs", experiment_id, seed, elapsed)

    # Aggregate all numeric keys
    all_keys: set = set()
    for r in per_seed_results:
        all_keys.update(k for k, v in r.items() if isinstance(v, (int, float)))

    aggregated: Dict[str, Any] = {}
    for key in sorted(all_keys):
        try:
            aggregated[key] = aggregate_seeds(per_seed_results, key)
        except (KeyError, TypeError):
            pass

    report = {
        "experiment_id": experiment_id,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "per_seed_results": per_seed_results,
        "aggregated": aggregated,
        "total_time_seconds": sum(per_seed_times),
        "mean_time_per_seed": float(sum(per_seed_times) / len(seeds)),
    }

    # Save to disk
    out_path = capsule.root / "report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
        
    if original_output_dir is not None:
        config.output_dir = original_output_dir
        
    log.info("[%s] Results saved to %s", experiment_id, out_path)

    return report


def compare_baselines(
    report: Dict[str, Any],
    primary_key: str,
    baseline_keys: List[str],
    metric: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Run paired t-tests comparing primary_key against each baseline_key.
    Applies Bonferroni correction.

    Parameters
    ----------
    report : dict from run_multi_seed
    primary_key : str
        The full metric key for the Z2 method (e.g., "z2_accuracy")
    baseline_keys : List[str]
        Full metric keys for baselines
    metric : str
        Human-readable name (for logging)

    Returns
    -------
    Dict with test results per baseline pair.
    """
    per_seed = report["per_seed_results"]
    primary_values = [r[primary_key] for r in per_seed]
    comparisons: Dict[str, Any] = {}
    p_values: List[float] = []

    for bkey in baseline_keys:
        baseline_values = [r[bkey] for r in per_seed]
        test = paired_ttest(primary_values, baseline_values)
        comparisons[bkey] = test
        p_values.append(test["p_value"])

    if p_values:
        corrections = bonferroni_correct(p_values, alpha)
        for i, bkey in enumerate(baseline_keys):
            comparisons[bkey]["bonferroni"] = corrections[i]

    return comparisons
