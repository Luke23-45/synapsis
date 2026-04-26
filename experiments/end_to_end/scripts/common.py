from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from experiments.end_to_end.data.collate import trajectory_collate_fn
from experiments.end_to_end.data.synapse_e2e_dataset import SynapseE2EDataset
from experiments.end_to_end.losses.combined_loss import LossConfig
from experiments.end_to_end.runtime import resolve_project_path
from experiments.end_to_end.training.trainer import compute_normalization_from_dataset
from synapse_arch.model import SynapseArchitectureConfig, SynapseEndToEndModel

log = logging.getLogger("e2e.scripts.common")


def load_checkpoint_bundle(
    checkpoint_path: str | Path,
    cfg: Any,
    device: torch.device,
) -> Dict[str, Any]:
    ckpt_path = resolve_project_path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    bundle = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" not in bundle or "epoch" not in bundle:
        raise ValueError(f"Checkpoint is missing required training state: {ckpt_path}")

    bundle["checkpoint_path"] = ckpt_path
    bundle["config"] = bundle.get("config", {})
    bundle["use_anchors"] = bundle["config"].get("use_anchors", cfg.ablation.use_anchors)
    bundle["use_topology"] = bundle["config"].get("use_topology", cfg.ablation.use_topology)
    return bundle


def build_eval_stack(
    cfg: Any,
    checkpoint_bundle: Dict[str, Any],
    device: torch.device,
) -> Tuple[SynapseEndToEndModel, DataLoader, SynapseE2EDataset, SynapseE2EDataset, LossConfig]:
    ckpt_config = checkpoint_bundle["config"]
    chunk_size = ckpt_config.get("action_chunk_size", cfg.model.action_chunk_size)

    train_ds = SynapseE2EDataset(
        dataset_path=str(resolve_project_path(cfg.data.train_lmdb)),
        chunk_size=chunk_size,
        min_history=cfg.data.min_history,
        proprio_noise=0.0,
        use_aug=False,
    )
    val_ds = SynapseE2EDataset(
        dataset_path=str(resolve_project_path(cfg.data.val_lmdb)),
        chunk_size=chunk_size,
        min_history=cfg.data.min_history,
        proprio_noise=0.0,
        use_aug=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=ckpt_config.get("batch_size", cfg.training.batch_size),
        shuffle=False,
        num_workers=cfg.training.get("num_workers", 0),
        collate_fn=trajectory_collate_fn,
        pin_memory=cfg.training.get("pin_memory", True) and device.type == "cuda",
        drop_last=False,
    )

    max_history_tokens = max(max(train_ds._episode_lengths), max(val_ds._episode_lengths))
    model_config = SynapseArchitectureConfig(
        input_dim=ckpt_config.get("input_dim", cfg.model.input_dim),
        action_dim=ckpt_config.get("action_dim", cfg.model.action_dim),
        action_chunk_size=chunk_size,
        hidden_dim=ckpt_config.get("hidden_dim", cfg.model.hidden_dim),
        d_model=ckpt_config.get("d_model", cfg.model.d_model),
        num_heads=ckpt_config.get("num_heads", cfg.model.num_heads),
        num_layers=ckpt_config.get("num_layers", cfg.model.num_layers),
        ffn_ratio=ckpt_config.get("ffn_ratio", cfg.model.ffn_ratio),
        dropout=0.0,
        K=ckpt_config.get("K", cfg.model.K),
        r=ckpt_config.get("r", cfg.model.r),
        lam=ckpt_config.get("lam", cfg.model.lam),
        Q=ckpt_config.get("Q", cfg.model.Q),
        k=ckpt_config.get("k", cfg.model.k),
        max_history_tokens=max_history_tokens,
    )
    model = SynapseEndToEndModel(model_config).to(device)
    model.load_state_dict(checkpoint_bundle["model_state_dict"])

    if "normalization" in checkpoint_bundle:
        model.normalized_lift.set_normalization(
            checkpoint_bundle["normalization"]["mu"].to(device),
            checkpoint_bundle["normalization"]["sigma"].to(device),
        )
    else:
        log.warning("Checkpoint missing normalization statistics; recomputing from training dataset.")
        norm_stats = compute_normalization_from_dataset(train_ds)
        model.normalized_lift.set_normalization(
            norm_stats.anchor_mu_tensor().to(device),
            norm_stats.anchor_sigma_tensor().to(device),
        )

    model.eval()
    return model, val_loader, train_ds, val_ds, LossConfig()
