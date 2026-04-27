#!/usr/bin/env python3
"""
SYNAPSE Phase 4 — Multi-Dataset Experiment Runner
===================================================

Trains all conditions across all configured datasets, evaluates,
runs statistical comparisons, and generates cross-domain reports.

Usage:
    python run_experiment.py --config configs/experiment/full.yaml
    python run_experiment.py --config configs/experiment/smoke.yaml
    python run_experiment.py --config configs/experiment/full.yaml --conditions A1_recent B_synapse
    python run_experiment.py --config configs/experiment/full.yaml --datasets pusht xarm_lift

This script orchestrates:
    1. Load dataset(s) via the adapter registry
    2. Per-dataset normalization statistics
    3. Per-dataset SYNAPSE feature caching
    4. Train each condition × dataset combination
    5. Evaluate on held-out test sets
    6. Cross-dataset statistical comparisons
    7. Publication-quality plots and reports
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config import (
    load_config,
    ExperimentConfig,
    Condition,
    DatasetSpec,
)
from src.core.normalization import (
    NormalizationStats,
    compute_normalization_stats_from_episodes,
)
from src.synapse.synapse_cache import (
    cache_synapse_features,
    load_cached_features,
)
from src.data.dataset import (
    split_episodes,
    create_dataloaders,
)
from src.data.adapters.base_adapter import RobotEpisode
from src.data.registry import create_adapter
from src.engine.train import Trainer
from src.engine.evaluate import Evaluator, ConditionEvaluation
from src.engine.metrics import compare_conditions, ComparisonResult
from src.reporting.visualize import (
    plot_learning_curves,
    plot_mse_comparison,
    plot_per_episode_scatter,
    plot_ablation_comparison,
)
from src.reporting.report import generate_json_report, generate_markdown_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)
_BASELINES_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def _load_dataset_episodes(
    ds_spec: DatasetSpec,
    config: ExperimentConfig,
    local_path_override: Optional[Path] = None,
) -> List[RobotEpisode]:
    """Load episodes from a single dataset using the adapter registry.
    """
    log.info("Loading dataset: %s (source=%s)", ds_spec.name, ds_spec.source)

    if str(_BASELINES_ROOT) not in sys.path:
        sys.path.insert(0, str(_BASELINES_ROOT))

    if ds_spec.source == "lerobot":
        from download_datasets import (
            DEFAULT_DATASET_ROOT,
            ensure_datasets_available,
            resolve_dataset_file,
        )

        dataset_root = Path(ds_spec.dataset_root) if ds_spec.dataset_root else DEFAULT_DATASET_ROOT
        if not dataset_root.is_absolute():
            dataset_root = (_BASELINES_ROOT / dataset_root).resolve()

        local_path = resolve_dataset_file(
            ds_spec.name,
            dataset_root=dataset_root,
            explicit_path=ds_spec.local_path,
        )
        if not local_path.is_absolute():
            local_path = (_BASELINES_ROOT / local_path).resolve()

        if not local_path.exists():
            log.info(
                "Dataset '%s' is missing locally. Verifying/downloading into %s.",
                ds_spec.name,
                dataset_root,
            )
            ensure_datasets_available([ds_spec.name], output_dir=dataset_root)
            local_path = resolve_dataset_file(ds_spec.name, dataset_root=dataset_root)
    elif ds_spec.source == "lmdb":
        path = local_path_override or ds_spec.local_path
        if path is None:
            raise ValueError(
                f"LMDB dataset '{ds_spec.name}' requires local_path or a split-specific path."
            )
        local_path = Path(path)
        if not local_path.is_absolute():
            local_path = (_PROJECT_ROOT / local_path).resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"LMDB dataset path not found: {local_path}")
    else:
        raise ValueError(
            f"Unsupported dataset source '{ds_spec.source}' for '{ds_spec.name}'."
        )

    adapter = create_adapter(
        ds_spec.name,
        local_path=local_path,
        max_episodes=ds_spec.max_episodes,
    )

    episodes = adapter.load_episodes()
    log.info(
        "Loaded %d episodes from '%s' (proprio_dim=%d, action_dim=%d)",
        len(episodes),
        ds_spec.name,
        adapter.proprio_dim,
        adapter.action_dim,
    )
    return episodes


def _load_dataset_splits(
    ds_spec: DatasetSpec,
    ds_config: ExperimentConfig,
) -> Tuple[List[RobotEpisode], List[RobotEpisode], List[RobotEpisode]]:
    """Load dataset splits, preserving explicit LMDB split files when provided."""
    if ds_spec.source == "lmdb" and ds_spec.train_path and ds_spec.val_path:
        train_eps = _load_dataset_episodes(
            ds_spec,
            ds_config,
            local_path_override=Path(ds_spec.train_path),
        )
        val_eps = _load_dataset_episodes(
            ds_spec,
            ds_config,
            local_path_override=Path(ds_spec.val_path),
        )
        if ds_spec.test_path:
            test_eps = _load_dataset_episodes(
                ds_spec,
                ds_config,
                local_path_override=Path(ds_spec.test_path),
            )
        else:
            log.info(
                "No explicit LMDB test split provided for %s; reusing validation episodes for test evaluation.",
                ds_spec.name,
            )
            test_eps = list(val_eps)
        return train_eps, val_eps, test_eps

    episodes = _load_dataset_episodes(ds_spec, ds_config)
    return split_episodes(
        episodes,
        train_ratio=ds_config.data.train_ratio,
        val_ratio=ds_config.data.val_ratio,
        seed=ds_config.seed,
    )


# ---------------------------------------------------------------------------
# Per-Dataset Experiment Pipeline
# ---------------------------------------------------------------------------

def _run_dataset_experiment(
    ds_spec: DatasetSpec,
    config: ExperimentConfig,
    conditions: List[Condition],
    output_base: Path,
    skip_cache: bool = False,
) -> Dict[str, dict]:
    """Run the full experiment for a single dataset.

    Returns
    -------
    dict mapping condition_name -> {
        "evaluation": ConditionEvaluation,
        "train_history": {"train_losses": [...], "val_losses": [...]},
    }
    """
    ds_name = ds_spec.name
    ds_output = output_base / f"dataset_{ds_name}"
    ds_output.mkdir(parents=True, exist_ok=True)

    # 1. Create dataset-specific config with correct dimensions
    ds_config = config.for_dataset(ds_spec)

    # 2. Load episodes / splits
    train_eps, val_eps, test_eps = _load_dataset_splits(ds_spec, ds_config)
    if not train_eps or not val_eps:
        log.error("Insufficient episodes loaded for '%s' — skipping.", ds_name)
        return {}

    # 4. Compute normalization from training split
    norm_stats = compute_normalization_stats_from_episodes(
        [{"proprio_history": ep.structured_history} for ep in train_eps]
    )
    norm_stats.save(ds_output / "normalization_stats.pt")
    action_norm_stats = compute_normalization_stats_from_episodes(
        [{"actions": ep.actions} for ep in train_eps],
        proprio_key="actions",
    )
    action_norm_stats.save(ds_output / "action_normalization_stats.pt")

    # 5. Cache SYNAPSE features only for the cached-feature implementation.
    needs_cached_features = any(
        c.uses_synapse and ds_config.uses_cached_synapse_features
        for c in conditions
    )
    if needs_cached_features and not skip_cache:
        for split_name, split_eps in [
            ("train", train_eps),
            ("val", val_eps),
            ("test", test_eps),
        ]:
            if not split_eps:
                continue

            episode_dicts = [
                {
                    "episode_id": ep.episode_id,
                    "proprio_history": ep.proprio_history,
                }
                for ep in split_eps
            ]

            cache_path = (
                ds_output / "synapse_features" / f"{split_name}.pt"
            )
            try:
                cache_synapse_features(
                    episode_dicts, ds_config, norm_stats, cache_path
                )
                cached = load_cached_features(cache_path)

                # Attach features to episodes
                for i, ep in enumerate(split_eps):
                    ep.synapse_anchors = cached["anchors"][i].numpy()
                    ep.synapse_topo = cached["topo"][i].numpy()
            except Exception as e:
                log.warning(
                    "SYNAPSE cache failed for %s/%s: %s. "
                    "Continuing with zero features.",
                    ds_name, split_name, e,
                )

    elif needs_cached_features and skip_cache:
        log.info(
            "Skipping SYNAPSE feature caching for %s because --skip-cache was set",
            ds_name,
        )

    # 6. Train and evaluate each condition
    results: Dict[str, dict] = {}

    for condition in conditions:
        cond_name = condition.value
        log.info(
            "═══ %s × %s ═══", ds_name.upper(), condition.label
        )

        cond_config = ExperimentConfig(
            condition=condition,
            synapse_implementation=ds_config.synapse_implementation,
            seed=ds_config.seed,
            synapse=ds_config.synapse,
            transformer=ds_config.transformer,
            training=ds_config.training,
            data=ds_config.data,
            stats=ds_config.stats,
            output_dir=ds_config.output_dir,
            experiment_name=ds_config.experiment_name,
            datasets=ds_config.datasets,
        )

        try:
            train_loader, val_loader, test_loader = create_dataloaders(
                train_eps,
                val_eps,
                test_eps,
                cond_config,
                norm_stats,
                action_norm_stats=action_norm_stats,
            )

            cond_dir = ds_output / f"condition_{cond_name}"
            trainer = Trainer(cond_config, train_loader, val_loader, cond_dir)
            train_state = trainer.train()

            evaluator = Evaluator(
                cond_config, test_loader, cond_dir / "evaluation"
            )
            evaluation = evaluator.evaluate_condition(trainer.model)

            results[cond_name] = {
                "evaluation": evaluation,
                "train_history": {
                    "train_losses": train_state.train_losses,
                    "val_losses": train_state.val_losses,
                },
            }

            log.info(
                "  %s × %s: mean_mse=%.6f ± %.6f",
                ds_name,
                cond_name,
                evaluation.mean_mse,
                evaluation.std_mse,
            )

        except Exception as e:
            log.error(
                "  %s × %s FAILED: %s", ds_name, cond_name, e,
                exc_info=True,
            )

    return results


# ---------------------------------------------------------------------------
# Cross-Dataset Reporting
# ---------------------------------------------------------------------------

def _generate_cross_domain_report(
    all_results: Dict[str, Dict[str, dict]],
    config: ExperimentConfig,
    output_base: Path,
) -> None:
    """Generate cross-domain comparison tables and plots.

    Parameters
    ----------
    all_results : dict
        Mapping dataset_name -> condition_name -> {evaluation, train_history}
    """
    report_dir = output_base / "cross_domain"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Build MSE matrix: dataset × condition
    datasets = sorted(all_results.keys())
    conditions = set()
    for ds_results in all_results.values():
        conditions.update(ds_results.keys())
    conditions = sorted(conditions)

    mse_matrix = {}
    for ds_name in datasets:
        mse_matrix[ds_name] = {}
        for cond_name in conditions:
            if cond_name in all_results[ds_name]:
                ev = all_results[ds_name][cond_name]["evaluation"]
                mse_matrix[ds_name][cond_name] = {
                    "mean": ev.mean_mse,
                    "std": ev.std_mse,
                }
            else:
                mse_matrix[ds_name][cond_name] = {"mean": float("nan"), "std": float("nan")}

    # Statistical comparisons (B vs A1, B vs A2) per dataset
    all_comparisons = {}
    for ds_name in datasets:
        ds_comparisons = []
        ds_results = all_results[ds_name]

        if "B_synapse" in ds_results and "A1_recent" in ds_results:
            try:
                comp = _compare_evals(
                    ds_results["A1_recent"]["evaluation"],
                    ds_results["B_synapse"]["evaluation"],
                    config,
                )
                ds_comparisons.append(comp)
            except Exception as e:
                log.warning("Comparison A1 vs B failed for %s: %s", ds_name, e)

        if "B_synapse" in ds_results and "A2_uniform" in ds_results:
            try:
                comp = _compare_evals(
                    ds_results["A2_uniform"]["evaluation"],
                    ds_results["B_synapse"]["evaluation"],
                    config,
                )
                ds_comparisons.append(comp)
            except Exception as e:
                log.warning("Comparison A2 vs B failed for %s: %s", ds_name, e)

        all_comparisons[ds_name] = ds_comparisons

    # Generate Markdown summary
    _write_cross_domain_markdown(
        mse_matrix, all_comparisons, datasets, conditions,
        report_dir / "cross_domain_summary.md",
    )

    # Generate JSON
    import json
    with open(report_dir / "mse_matrix.json", "w") as f:
        json.dump(mse_matrix, f, indent=2, default=str)

    # Per-dataset reports
    for ds_name in datasets:
        ds_dir = report_dir / ds_name
        ds_dir.mkdir(exist_ok=True)

        ds_results = all_results[ds_name]
        evaluations = {
            k: v["evaluation"] for k, v in ds_results.items()
        }

        if evaluations:
            # MSE comparison plot
            cond_means = {k: e.mean_mse for k, e in evaluations.items()}
            cond_stds = {k: e.std_mse for k, e in evaluations.items()}
            try:
                plot_mse_comparison(
                    cond_means, cond_stds,
                    ds_dir / f"{ds_name}_mse_comparison.pdf",
                )
            except Exception as e:
                log.warning("Plot failed for %s: %s", ds_name, e)

            # Learning curves
            train_losses = {
                k: v["train_history"]["train_losses"]
                for k, v in ds_results.items()
            }
            val_losses = {
                k: v["train_history"]["val_losses"]
                for k, v in ds_results.items()
            }
            try:
                plot_learning_curves(
                    train_losses, val_losses,
                    ds_dir / f"{ds_name}_learning_curves.pdf",
                )
            except Exception as e:
                log.warning("Learning curve plot failed for %s: %s", ds_name, e)

    log.info("Cross-domain reports saved to %s", report_dir)


def _write_cross_domain_markdown(
    mse_matrix: dict,
    comparisons: dict,
    datasets: List[str],
    conditions: List[str],
    output_path: Path,
) -> None:
    """Write a Markdown summary table."""
    lines = [
        "# SYNAPSE Phase 4 — Cross-Domain Results\n",
        "",
        "## MSE Comparison Matrix (mean ± std)\n",
        "",
    ]

    # Header
    header = "| Dataset | " + " | ".join(conditions) + " |"
    sep = "|---------|" + "|".join(["--------"] * len(conditions)) + "|"
    lines.extend([header, sep])

    for ds_name in datasets:
        row = f"| **{ds_name}** |"
        for cond in conditions:
            entry = mse_matrix[ds_name][cond]
            if np.isnan(entry["mean"]):
                row += " — |"
            else:
                row += f" {entry['mean']:.4f} ± {entry['std']:.4f} |"
        lines.append(row)

    lines.append("")
    lines.append("## Statistical Comparisons (Welch's t-test)\n")

    for ds_name, comps in comparisons.items():
        if comps:
            lines.append(f"\n### {ds_name}\n")
            for comp in comps:
                sig = "✅ Significant" if comp.p_value < 0.05 else "❌ Not significant"
                lines.append(
                    f"- **{comp.condition_a} vs {comp.condition_b}**: "
                    f"p={comp.p_value:.4f}, d={comp.cohens_d:.3f} ({sig})"
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def _compare_evals(
    eval_a: ConditionEvaluation,
    eval_b: ConditionEvaluation,
    config: ExperimentConfig,
) -> ComparisonResult:
    """Compare two condition evaluations statistically."""
    n = min(len(eval_a.per_episode_mse), len(eval_b.per_episode_mse))
    return compare_conditions(
        eval_a.per_episode_mse[:n],
        eval_b.per_episode_mse[:n],
        metric_name="action_mse",
        condition_a=eval_a.condition.value,
        condition_b=eval_b.condition.value,
        alpha=config.stats.significance_level,
        n_bootstrap=config.stats.num_bootstrap_samples,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SYNAPSE Phase 4: Multi-dataset cross-domain experiment"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--conditions", type=str, nargs="+", default=None,
        help="Override conditions (e.g., A1_recent B_synapse)",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None,
        help="Override datasets to run (e.g., pusht xarm_lift)",
    )
    parser.add_argument(
        "--skip-cache", action="store_true",
        help="Skip SYNAPSE feature caching (use existing or zeros)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    log.info("Loaded config: %s", args.config)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = (
        Path(config.output_dir) / f"{timestamp}_{config.experiment_name}"
    )
    output_base.mkdir(parents=True, exist_ok=True)
    config.save_yaml(output_base / "config_snapshot.yaml")

    # Determine conditions
    if args.conditions:
        conditions = [Condition(c) for c in args.conditions]
    else:
        conditions = [
            Condition.A1_RECENT,
            Condition.A2_UNIFORM,
            Condition.B_SYNAPSE,
        ]

    # Determine datasets
    if args.datasets:
        ds_specs = [
            ds for ds in config.datasets if ds.name in args.datasets
        ]
        if not ds_specs:
            log.error(
                "None of the requested datasets found in config: %s",
                args.datasets,
            )
            sys.exit(1)
    else:
        ds_specs = list(config.datasets)

    log.info(
        "═══════════════════════════════════════════════════════════"
    )
    log.info(
        "SYNAPSE Phase 4 Experiment: %d datasets × %d conditions",
        len(ds_specs),
        len(conditions),
    )
    log.info(
        "  Datasets: %s", [ds.name for ds in ds_specs]
    )
    log.info(
        "  Conditions: %s", [c.value for c in conditions]
    )
    log.info(
        "═══════════════════════════════════════════════════════════"
    )

    # Run experiment for each dataset
    all_results: Dict[str, Dict[str, dict]] = {}
    total_start = time.time()

    for ds_spec in ds_specs:
        log.info(
            "\n▓▓▓ DATASET: %s ▓▓▓\n", ds_spec.name.upper()
        )
        ds_start = time.time()

        results = _run_dataset_experiment(
            ds_spec, config, conditions, output_base, skip_cache=args.skip_cache
        )

        ds_elapsed = time.time() - ds_start
        log.info(
            "Dataset '%s' complete in %.1fs (%d conditions)",
            ds_spec.name,
            ds_elapsed,
            len(results),
        )

        if results:
            all_results[ds_spec.name] = results

    total_elapsed = time.time() - total_start

    # Generate cross-domain reports
    if all_results:
        _generate_cross_domain_report(all_results, config, output_base)

    log.info(
        "═══════════════════════════════════════════════════════════"
    )
    log.info(
        "EXPERIMENT COMPLETE: %d datasets, %.1fs total",
        len(all_results),
        total_elapsed,
    )
    log.info("Output directory: %s", output_base)
    log.info(
        "═══════════════════════════════════════════════════════════"
    )


if __name__ == "__main__":
    main()
