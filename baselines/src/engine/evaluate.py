"""
Evaluation Engine — M1 Experiment
==================================

Evaluation protocol (§5 of PLAN.md):
    1. Load best checkpoint for each condition
    2. Run on held-out test set
    3. Collect per-episode MSE for each condition
    4. Compute paired differences (same episodes, different conditions)
    5. Run Welch's t-test on paired differences
    6. Compute Cohen's d and 95% CI
    7. Stratify by episode length and phase-transition density
    8. Run autoregressive rollout evaluation
    9. Generate comparison tables and plots
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.core.config import ExperimentConfig, Condition
from src.planner.planner import RoboticsPlannerBase
from src.planner.planner_synapse import create_planner
from src.data.dataset import DataLoader
from .metrics import (
    ComparisonResult,
    compare_conditions,
    stratify_by_episode_length,
    stratify_by_phase_density,
)
from .rollout import (
    RolloutResult,
    rollout_evaluate_dataset,
    aggregate_rollout_results,
)

log = logging.getLogger(__name__)


@dataclass
class ConditionEvaluation:
    """Evaluation results for a single condition.

    Attributes
    ----------
    condition : Condition
    per_episode_mse : np.ndarray, shape (N,)
        Action MSE for each test episode.
    per_episode_cosine : np.ndarray, shape (N,)
        Action cosine similarity for each test episode.
    per_episode_phase_acc : np.ndarray, shape (N,)
        Phase prediction accuracy for each test episode.
    episode_lengths : np.ndarray, shape (N,)
    phase_boundaries : np.ndarray, shape (N,)
        Number of phase boundaries per episode.
    mean_mse : float
    std_mse : float
    mean_cosine : float
    mean_phase_acc : float
    rollout : dict or None
        Aggregated rollout results (if computed).
    """

    condition: Condition
    per_episode_mse: np.ndarray
    per_episode_cosine: np.ndarray
    per_episode_phase_acc: np.ndarray
    per_episode_mae: np.ndarray
    per_episode_smooth_l1: np.ndarray
    episode_lengths: np.ndarray
    phase_boundaries: np.ndarray
    mean_mse: float = 0.0
    std_mse: float = 0.0
    mean_cosine: float = 0.0
    mean_phase_acc: float = 0.0
    mean_mae: float = 0.0
    mean_smooth_l1: float = 0.0
    rollout: Optional[Dict] = None

    def __post_init__(self):
        if len(self.per_episode_mse) > 0:
            self.mean_mse = float(np.mean(self.per_episode_mse))
            self.std_mse = float(np.std(self.per_episode_mse, ddof=1))
            self.mean_cosine = float(np.mean(self.per_episode_cosine))
            self.mean_phase_acc = float(np.mean(self.per_episode_phase_acc))
            self.mean_mae = float(np.mean(self.per_episode_mae))
            self.mean_smooth_l1 = float(np.mean(self.per_episode_smooth_l1))


class Evaluator:
    """Evaluation engine for the M1 experiment.

    Parameters
    ----------
    config : ExperimentConfig
    test_loader : DataLoader
    output_dir : Path
    """

    def __init__(
        self,
        config: ExperimentConfig,
        test_loader: DataLoader,
        output_dir: Path,
    ) -> None:
        self.config = config
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @torch.no_grad()
    def evaluate_condition(
        self,
        model: RoboticsPlannerBase,
        compute_rollout: bool = True,
        max_rollout_episodes: int = 50,
    ) -> ConditionEvaluation:
        """Evaluate a single trained model on the test set.

        Parameters
        ----------
        model : RoboticsPlannerBase
            Trained planner model.
        compute_rollout : bool
            Whether to run autoregressive rollout evaluation.
        max_rollout_episodes : int
            Maximum episodes for rollout (compute budget).

        Returns
        -------
        ConditionEvaluation
        """
        model.eval()
        model.to(self.device)

        episode_metrics: Dict[int, Dict[str, List[float] | int]] = {}

        for batch in self.test_loader:
            batch_device = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if self.config.uses_end_to_end_synapse and hasattr(model, "forward_deploy"):
                deploy_out = model.forward_deploy(batch_device)
                pred_actions = deploy_out.pred_actions
            else:
                pred_actions = model(batch_device)
            gt_actions = batch_device["action_chunk"]  # (B, K_act, D_act)

            # Per-sample MSE
            sample_mse = F.mse_loss(
                pred_actions, gt_actions, reduction="none"
            ).mean(dim=(1, 2))  # (B,)

            # Per-sample cosine similarity
            pred_flat = pred_actions.reshape(pred_actions.shape[0], -1)
            gt_flat = gt_actions.reshape(gt_actions.shape[0], -1)
            sample_cosine = F.cosine_similarity(pred_flat, gt_flat, dim=1)  # (B,)
            sample_mae = F.l1_loss(
                pred_actions, gt_actions, reduction="none"
            ).mean(dim=(1, 2))
            sample_smooth_l1 = F.smooth_l1_loss(
                pred_actions, gt_actions, reduction="none"
            ).mean(dim=(1, 2))

            B = sample_mse.shape[0]
            for b in range(B):
                episode_idx = int(batch["episode_idx"][b].item())
                episode_state = episode_metrics.setdefault(
                    episode_idx,
                    {
                        "mse": [],
                        "cosine": [],
                        "mae": [],
                        "smooth_l1": [],
                        "phase_acc": [],
                        "episode_length": int(batch["episode_length"][b].item()),
                        "phase_boundaries": 0,
                    },
                )
                episode_state["mse"].append(float(sample_mse[b].item()))
                episode_state["cosine"].append(float(sample_cosine[b].item()))
                episode_state["mae"].append(float(sample_mae[b].item()))
                episode_state["smooth_l1"].append(float(sample_smooth_l1[b].item()))
                episode_state["phase_acc"].append(0.0)

        ordered_episode_indices = sorted(episode_metrics)
        per_episode_mse = np.array(
            [np.mean(episode_metrics[idx]["mse"]) for idx in ordered_episode_indices],
            dtype=np.float64,
        )
        per_episode_cosine = np.array(
            [np.mean(episode_metrics[idx]["cosine"]) for idx in ordered_episode_indices],
            dtype=np.float64,
        )
        per_episode_mae = np.array(
            [np.mean(episode_metrics[idx]["mae"]) for idx in ordered_episode_indices],
            dtype=np.float64,
        )
        per_episode_smooth_l1 = np.array(
            [np.mean(episode_metrics[idx]["smooth_l1"]) for idx in ordered_episode_indices],
            dtype=np.float64,
        )
        phase_acc_arr = np.array(
            [np.mean(episode_metrics[idx]["phase_acc"]) for idx in ordered_episode_indices],
            dtype=np.float64,
        )
        lengths_arr = np.array(
            [episode_metrics[idx]["episode_length"] for idx in ordered_episode_indices],
            dtype=np.float64,
        )
        boundaries_arr = np.array(
            [episode_metrics[idx]["phase_boundaries"] for idx in ordered_episode_indices],
            dtype=np.float64,
        )

        # Rollout evaluation
        rollout_results = None
        if compute_rollout:
            try:
                rollout_dict = rollout_evaluate_dataset(
                    model,
                    self.test_loader,
                    self.config,
                    n_steps=self.config.stats.rollout_steps,
                    max_episodes=max_rollout_episodes,
                    dump_topology_dir=self.output_dir / "topology_dumps",
                )
                rollout_results = aggregate_rollout_results(rollout_dict)
            except Exception as e:
                log.error("Rollout evaluation failed: %s", e)

        return ConditionEvaluation(
            condition=self.config.condition,
            per_episode_mse=per_episode_mse,
            per_episode_cosine=per_episode_cosine,
            per_episode_phase_acc=phase_acc_arr,
            per_episode_mae=per_episode_mae,
            per_episode_smooth_l1=per_episode_smooth_l1,
            episode_lengths=lengths_arr,
            phase_boundaries=boundaries_arr,
            rollout=rollout_results,
        )

    def compare_conditions(
        self,
        eval_a: ConditionEvaluation,
        eval_b: ConditionEvaluation,
        metric_name: str = "action_mse",
        num_comparisons: int = 1,
    ) -> ComparisonResult:
        """Statistically compare two conditions.

        Parameters
        ----------
        eval_a, eval_b : ConditionEvaluation
        metric_name : str
        num_comparisons : int
            For Bonferroni correction.

        Returns
        -------
        ComparisonResult
        """
        if metric_name == "action_mse":
            scores_a = eval_a.per_episode_mse
            scores_b = eval_b.per_episode_mse
        elif metric_name == "cosine_similarity":
            scores_a = eval_a.per_episode_cosine
            scores_b = eval_b.per_episode_cosine
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

        # Ensure same number of episodes (paired comparison)
        n = min(len(scores_a), len(scores_b))
        if n < 2:
            log.warning(
                "Insufficient samples for comparison: %d", n
            )
            # Return a default result
            return ComparisonResult(
                metric_name=metric_name,
                condition_a=eval_a.condition.value,
                condition_b=eval_b.condition.value,
                mean_a=float(np.mean(scores_a)) if len(scores_a) > 0 else 0.0,
                std_a=0.0,
                mean_b=float(np.mean(scores_b)) if len(scores_b) > 0 else 0.0,
                std_b=0.0,
                mean_diff=0.0,
                cohens_d=0.0,
                d_ci_lower=0.0,
                d_ci_upper=0.0,
                t_statistic=0.0,
                p_value=1.0,
                welch_significant=False,
                shapiro_p=1.0,
                is_normal=True,
                wilcoxon_p=None,
                wilcoxon_significant=None,
                bonferroni_alpha=0.05 / max(1, num_comparisons),
                significant_after_correction=False,
            )

        return compare_conditions(
            scores_a[:n],
            scores_b[:n],
            metric_name=metric_name,
            condition_a=eval_a.condition.value,
            condition_b=eval_b.condition.value,
            alpha=self.config.stats.significance_level,
            n_bootstrap=self.config.stats.num_bootstrap_samples,
            num_comparisons=num_comparisons,
        )

    def stratified_analysis(
        self,
        eval_result: ConditionEvaluation,
    ) -> Dict[str, Dict[str, float]]:
        """Stratify evaluation by episode length and phase density.

        Returns
        -------
        dict with keys 'by_length', 'by_phase_density',
            each mapping stratum_label → {mean_mse, std_mse, count}
        """
        by_length = {}
        if len(eval_result.per_episode_mse) > 0:
            length_strata = stratify_by_episode_length(
                eval_result.episode_lengths,
                eval_result.per_episode_mse,
            )
            for label, scores in length_strata.items():
                if len(scores) > 0:
                    by_length[label] = {
                        "mean_mse": float(np.mean(scores)),
                        "std_mse": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                        "count": len(scores),
                    }

        by_density = {}
        if len(eval_result.per_episode_mse) > 0 and len(eval_result.phase_boundaries) > 0:
            density_strata = stratify_by_phase_density(
                eval_result.phase_boundaries,
                eval_result.per_episode_mse,
                eval_result.episode_lengths,
            )
            for label, scores in density_strata.items():
                if len(scores) > 0:
                    by_density[label] = {
                        "mean_mse": float(np.mean(scores)),
                        "std_mse": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                        "count": len(scores),
                    }

        return {"by_length": by_length, "by_phase_density": by_density}
