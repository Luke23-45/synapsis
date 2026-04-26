"""
Checkpoint save/load for Z2 end-to-end training.

Each checkpoint captures the full training state:
  - Model weights
  - EMA shadow weights
  - Optimizer + scheduler state
  - Configuration, metrics, normalization stats, split indices

Compatible with synapse_arch.api.load_model() for inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .ema import EMAModel

log = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside each checkpoint."""

    epoch: int
    metrics: Dict[str, float]
    config: Dict[str, Any]
    seed: int


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    ema: EMAModel,
    metadata: CheckpointMetadata,
    normalization: Optional[Dict[str, torch.Tensor]] = None,
    split_indices: Optional[Dict[str, list]] = None,
) -> None:
    """Save a complete training checkpoint.

    Parameters
    ----------
    path : Path
        File path for the checkpoint (e.g., checkpoints/best.pt).
    model : nn.Module
        The model to save.
    optimizer : Optimizer
        Optimizer state for resumption.
    scheduler : LRScheduler
        Scheduler state for resumption.
    ema : EMAModel
        EMA state for evaluation.
    metadata : CheckpointMetadata
        Epoch, metrics, config, seed.
    normalization : dict, optional
        {"mu": Tensor, "sigma": Tensor} for NormalizedLift.
    split_indices : dict, optional
        {"train": [...], "val": [...], "test": [...]} episode indices.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": metadata.epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "config": metadata.config,
        "metrics": metadata.metrics,
        "seed": metadata.seed,
    }

    if normalization is not None:
        checkpoint["normalization"] = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in normalization.items()
        }

    if split_indices is not None:
        checkpoint["split_indices"] = split_indices

    torch.save(checkpoint, path)
    log.info(
        "Checkpoint saved: %s (epoch=%d, loss=%.6f)",
        path.name,
        metadata.epoch,
        metadata.metrics.get("val_loss", float("nan")),
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    ema: Optional[EMAModel] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint and restore model/optimizer/scheduler/ema state.

    Parameters
    ----------
    path : Path
        Checkpoint file path.
    model : nn.Module
        Model to restore weights into.
    optimizer : Optimizer, optional
        If provided, optimizer state is restored (for resuming training).
    scheduler : LRScheduler, optional
        If provided, scheduler state is restored.
    ema : EMAModel, optional
        If provided, EMA state is restored.
    map_location : str
        Device to map checkpoint tensors to.

    Returns
    -------
    Dict[str, Any]
        The full checkpoint dict (for accessing metadata, config, etc.).
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    log.info("Model weights restored from %s (epoch=%d)", path.name, checkpoint["epoch"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        log.info("Optimizer state restored")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        log.info("Scheduler state restored")

    if ema is not None and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])
        log.info("EMA state restored")

    return checkpoint
