"""
Evaluation for Z2 end-to-end training.

This module computes publication-facing metrics for both the relaxed
training path and the exact deployment path. It also records evaluation
coverage so failed deploy batches or empty loaders are visible in the
result payload instead of being silently ignored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..losses.combined_loss import LossConfig, combined_loss

log = logging.getLogger(__name__)


@dataclass
class EvaluationAccumulator:
    total_loss: float = 0.0
    total_action_mse: float = 0.0
    total_action_mae: float = 0.0
    total_anchor_util: float = 0.0
    total_batches: int = 0
    total_samples: int = 0
    skipped_batches: int = 0
    failed_batches: int = 0


def _autocast_context(device: torch.device, enabled: bool):
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type=device.type, enabled=enabled, dtype=amp_dtype)


@torch.no_grad()
def evaluate_train_path(
    model: nn.Module,
    loader: DataLoader,
    loss_config: LossConfig,
    epoch: int,
    device: torch.device,
    amp_enabled: bool = False,
    use_anchors: bool = True,
    use_topology: bool = True,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.eval()
    acc = EvaluationAccumulator()

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if not batch:
            acc.skipped_batches += 1
            continue

        batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

        with _autocast_context(device, amp_enabled):
            output = model.forward_train(batch, use_anchors=use_anchors, use_topology=use_topology)
            loss, loss_dict = combined_loss(output, batch, epoch, loss_config)

        pred = output.pred_actions
        gt = batch["ground_truth_actions"]
        batch_size = int(gt.shape[0])

        acc.total_loss += float(loss_dict["loss_total"])
        acc.total_action_mse += float(torch.nn.functional.mse_loss(pred, gt).item())
        acc.total_action_mae += float((pred - gt).abs().mean().item())
        active_count = (output.y_star > 0.5).float().sum(dim=1).mean().item()
        K = model.relaxed_selector.K if hasattr(model, "relaxed_selector") else 10
        acc.total_anchor_util += active_count / max(K, 1)
        acc.total_batches += 1
        acc.total_samples += batch_size

    if acc.total_batches == 0:
        return {
            "val_loss": float("nan"),
            "val_action_mse": float("nan"),
            "val_action_mae": float("nan"),
            "val_anchor_util": float("nan"),
            "val_num_batches": 0.0,
            "val_num_samples": 0.0,
            "val_skipped_batches": float(acc.skipped_batches),
        }

    return {
        "val_loss": acc.total_loss / acc.total_batches,
        "val_action_mse": acc.total_action_mse / acc.total_batches,
        "val_action_mae": acc.total_action_mae / acc.total_batches,
        "val_anchor_util": acc.total_anchor_util / acc.total_batches,
        "val_num_batches": float(acc.total_batches),
        "val_num_samples": float(acc.total_samples),
        "val_skipped_batches": float(acc.skipped_batches),
    }


@torch.no_grad()
def evaluate_deploy_path(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    use_anchors: bool = True,
    use_topology: bool = True,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_batches = 0
    total_samples = 0
    skipped_batches = 0
    failed_batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if not batch:
            skipped_batches += 1
            continue

        batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

        try:
            with _autocast_context(device, amp_enabled):
                output = model.forward_deploy(batch, use_anchors=use_anchors, use_topology=use_topology)
            pred = output.pred_actions
            gt = batch["ground_truth_actions"]

            total_mse += float(torch.nn.functional.mse_loss(pred, gt).item())
            total_mae += float((pred - gt).abs().mean().item())
            total_batches += 1
            total_samples += int(gt.shape[0])
        except Exception as exc:
            failed_batches += 1
            log.warning("Deploy path evaluation failed on batch %d: %s", batch_idx, exc)

    if total_batches == 0:
        return {
            "deploy_mse": float("nan"),
            "deploy_mae": float("nan"),
            "deploy_num_batches": 0.0,
            "deploy_num_samples": 0.0,
            "deploy_failed_batches": float(failed_batches),
            "deploy_skipped_batches": float(skipped_batches),
        }

    return {
        "deploy_mse": total_mse / total_batches,
        "deploy_mae": total_mae / total_batches,
        "deploy_num_batches": float(total_batches),
        "deploy_num_samples": float(total_samples),
        "deploy_failed_batches": float(failed_batches),
        "deploy_skipped_batches": float(skipped_batches),
    }


def compute_train_deploy_gap(
    train_metrics: Dict[str, float],
    deploy_metrics: Dict[str, float],
) -> Dict[str, float]:
    train_mse = train_metrics.get("val_action_mse", float("nan"))
    deploy_mse = deploy_metrics.get("deploy_mse", float("nan"))

    if np.isnan(train_mse) or np.isnan(deploy_mse) or train_mse <= 0:
        return {
            "gap_mse_absolute": float("nan"),
            "gap_mse_relative": float("nan"),
        }

    gap_abs = abs(train_mse - deploy_mse)
    gap_rel = gap_abs / train_mse
    return {
        "gap_mse_absolute": float(gap_abs),
        "gap_mse_relative": float(gap_rel),
    }


def full_evaluation(
    model: nn.Module,
    loader: DataLoader,
    loss_config: LossConfig,
    epoch: int,
    device: torch.device,
    amp_enabled: bool = False,
    use_anchors: bool = True,
    use_topology: bool = True,
    max_batches: int | None = None,
) -> Dict[str, float]:
    train_metrics = evaluate_train_path(
        model,
        loader,
        loss_config,
        epoch,
        device,
        amp_enabled=amp_enabled,
        use_anchors=use_anchors,
        use_topology=use_topology,
        max_batches=max_batches,
    )
    deploy_metrics = evaluate_deploy_path(
        model,
        loader,
        device,
        amp_enabled=amp_enabled,
        use_anchors=use_anchors,
        use_topology=use_topology,
        max_batches=max_batches,
    )
    gap_metrics = compute_train_deploy_gap(train_metrics, deploy_metrics)

    combined: Dict[str, float] = {}
    combined.update(train_metrics)
    combined.update(deploy_metrics)
    combined.update(gap_metrics)
    return combined
