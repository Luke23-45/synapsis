"""
Main training loop for Z2 end-to-end training.

Pure PyTorch implementation (not Lightning) for full control over:
  - Bifurcated forward (forward_train vs forward_deploy)
  - OSQP solver integration (CPU-bound, needs careful batching)
  - EMA model management
  - Custom logging matching Phase 2 infrastructure

Architecture:
  1. Load data via SynapseE2EDataset (reads LMDB directly)
  2. Separate train/val LMDB files (500 train, 50 val episodes)
  3. EpisodeAwareSampler for LMDB cache locality
  4. Compute normalization statistics from training data
  5. Build model, optimizer, scheduler, EMA
  6. Train loop with gradient clipping, periodic evaluation, checkpointing
  7. Final evaluation with train-deploy gap analysis
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from synapse_arch.model import SynapseArchitectureConfig, SynapseEndToEndModel

from ..data.collate import trajectory_collate_fn
from ..data.normalization import NormalizationStats, compute_normalization_stats
from ..runtime import build_run_artifacts, relativize_to_project, resolve_project_path, save_json
from ..data.samplers import EpisodeAwareSampler
from ..data.synapse_e2e_dataset import SynapseE2EDataset
from ..losses.combined_loss import LossConfig, combined_loss
from .checkpointer import CheckpointMetadata, load_checkpoint, save_checkpoint
from .ema import EMAModel
from .evaluator import evaluate_train_path, full_evaluation
from .scheduler import create_cosine_warmup_scheduler

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Complete training configuration."""

    # Model
    input_dim: int = 22
    action_dim: int = 8
    action_chunk_size: int = 10
    hidden_dim: int = 64
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 3
    ffn_ratio: int = 4
    dropout: float = 0.1
    K: int = 10
    r: int = 2
    lam: float = 1.0
    Q: int = 1
    k: int = 8

    # Training
    batch_size: int = 16
    epochs: int = 300
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    warmup_epochs: int = 15
    min_lr: float = 1e-6
    gradient_clip: float = 1.0
    ema_decay: float = 0.999
    early_stopping_patience: int = 40
    val_check_interval: int = 5

    # Losses
    action_weight: float = 1.0
    sparsity_weight: float = 0.01
    topology_reg_weight: float = 0.001
    action_beta: float = 0.5
    aux_ramp_start: int = 10
    aux_ramp_end: int = 30

    # Data
    proprio_noise: float = 0.005
    use_aug: bool = True
    min_history: int = 2

    # Ablation flags
    use_anchors: bool = True
    use_topology: bool = True

    # Performance / Optimization
    use_amp: bool = False
    compile_model: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None
    max_analysis_batches: Optional[int] = None

    # Logging / output
    output_dir: str = "experiments/outputs/end_to_end"
    log_interval: int = 10
    save_interval: int = 50
    manifest_name: str = "run_manifest.json"
    resolved_config_name: str = "resolved_config.yaml"
    seed: int = 42
    use_wandb: bool = False
    wandb_project: str = "SYNAPSE"
    wandb_mode: str = "offline"
    wandb_run_name: Optional[str] = None
    analysis_enabled: bool = True
    anchor_heatmap_samples: int = 20
    topology_tsne_max_samples: int = 2000
    topology_tsne_perplexity: int = 30
    topology_tsne_random_seed: int = 42

    def to_model_config(self, max_history_tokens: int) -> SynapseArchitectureConfig:
        """Convert to SynapseArchitectureConfig."""
        return SynapseArchitectureConfig(
            input_dim=self.input_dim,
            action_dim=self.action_dim,
            action_chunk_size=self.action_chunk_size,
            hidden_dim=self.hidden_dim,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ffn_ratio=self.ffn_ratio,
            dropout=self.dropout,
            K=self.K,
            r=self.r,
            lam=self.lam,
            Q=self.Q,
            k=self.k,
            max_history_tokens=max_history_tokens,
        )

    def to_loss_config(self) -> LossConfig:
        """Convert to LossConfig."""
        return LossConfig(
            action_weight=self.action_weight,
            sparsity_weight=self.sparsity_weight,
            topology_reg_weight=self.topology_reg_weight,
            action_beta=self.action_beta,
            aux_ramp_start=self.aux_ramp_start,
            aux_ramp_end=self.aux_ramp_end,
        )


# ──────────────────────────────────────────────────────────────────────
# Normalization from SynapseE2EDataset
# ──────────────────────────────────────────────────────────────────────

def compute_normalization_from_dataset(
    dataset: SynapseE2EDataset,
    max_episodes: int = 0,
) -> NormalizationStats:
    """Compute normalization statistics from a SynapseE2EDataset.

    Materializes episodes from the LMDB reader to compute per-feature
    mean/std for states and anchor vectors.

    Parameters
    ----------
    dataset : SynapseE2EDataset
        The training dataset (reads from LMDB).
    max_episodes : int
        If > 0, use only this many episodes (for speed).

    Returns
    -------
    NormalizationStats
    """
    reader = dataset.expert_reader
    num_episodes = reader.get_num_episodes()
    if max_episodes > 0:
        num_episodes = min(num_episodes, max_episodes)

    episode_dicts = []
    for ep_idx in range(num_episodes):
        ep_meta = reader.episode_metadata[ep_idx]

        def get_mod(name: str) -> np.ndarray:
            meta = ep_meta["modalities"][name]
            return reader._get_full_modality_array(
                key=meta["key"],
                compression=meta["compression"],
                dtype_str=meta["dtype"],
                shape_list=tuple(meta["shape"]),
            )

        episode_dicts.append({
            "states": get_mod("proprio"),
            "actions": get_mod("actions"),
        })

    return compute_normalization_stats(episode_dicts)


# ──────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────

def train(
    config: TrainingConfig,
    train_lmdb_path: str,
    val_lmdb_path: str,
    device: Optional[torch.device] = None,
    raw_cfg: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute the full Z2 end-to-end training pipeline.

    Parameters
    ----------
    config : TrainingConfig
        Complete training configuration.
    train_lmdb_path : str
        Path to the training LMDB dataset (e.g., 500 episodes).
    val_lmdb_path : str
        Path to the validation LMDB dataset (e.g., 50 episodes).
    device : torch.device, optional
        Device to train on. Defaults to CUDA if available, else CPU.

    Returns
    -------
    Dict[str, Any]
        Final test metrics and training history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_lmdb_path = str(resolve_project_path(train_lmdb_path))
    val_lmdb_path = str(resolve_project_path(val_lmdb_path))
    artifacts = build_run_artifacts(
        config.output_dir,
        config.seed,
        manifest_name=config.manifest_name,
        resolved_config_name=config.resolved_config_name,
    )
    run_dir = artifacts["run_dir"]
    ckpt_dir = artifacts["checkpoint_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Z2 End-to-End Training")
    log.info("  Device:     %s", device)
    log.info("  Output:     %s", run_dir)
    log.info("  Seed:       %d", config.seed)
    log.info("  Train LMDB: %s", train_lmdb_path)
    log.info("  Val LMDB:   %s", val_lmdb_path)
    log.info("=" * 70)

    if raw_cfg is not None:
        OmegaConf.save(raw_cfg, artifacts["resolved_config"])
        log.info("Resolved config saved to %s", artifacts["resolved_config"])

    if config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or f"z2_e2e_seed_{config.seed}",
                mode=config.wandb_mode,
                dir=str(run_dir),
                config=asdict(config),
            )
            log.info("WandB initialized in %s mode.", config.wandb_mode)
        except ImportError:
            log.warning("wandb is not installed. Disabling WandB integration.")
            config.use_wandb = False

    # ── 1. Create datasets (reads directly from LMDB) ─────────────
    train_ds = SynapseE2EDataset(
        dataset_path=train_lmdb_path,
        chunk_size=config.action_chunk_size,
        min_history=config.min_history,
        proprio_noise=config.proprio_noise,
        use_aug=config.use_aug,
    )

    val_ds = SynapseE2EDataset(
        dataset_path=val_lmdb_path,
        chunk_size=config.action_chunk_size,
        min_history=config.min_history,
        proprio_noise=0.0,    # No noise for validation
        use_aug=False,         # No augmentation for validation
    )

    log.info(
        "Datasets: train=%d samples (%d episodes), val=%d samples (%d episodes)",
        len(train_ds), train_ds.expert_reader.get_num_episodes(),
        len(val_ds), val_ds.expert_reader.get_num_episodes(),
    )

    # ── 2. Compute normalization from training data ───────────────
    norm_stats = compute_normalization_from_dataset(train_ds)

    # ── 3. Create data loaders with EpisodeAwareSampler ───────────
    train_sampler = EpisodeAwareSampler(
        train_ds, shuffle=True, seed=config.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        collate_fn=trajectory_collate_fn,
        drop_last=False,
        pin_memory=config.pin_memory and device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=trajectory_collate_fn,
        drop_last=False,
        pin_memory=config.pin_memory and device.type == "cuda",
    )

    # ── 4. Model ──────────────────────────────────────────────────
    # max_history_tokens must accommodate the longest possible history
    # across BOTH train and val datasets
    max_T_train = max(train_ds._episode_lengths)
    max_T_val = max(val_ds._episode_lengths)
    max_T = max(max_T_train, max_T_val)
    log.info("Max episode length: train=%d, val=%d -> max_history_tokens=%d",
             max_T_train, max_T_val, max_T)

    model_config = config.to_model_config(max_history_tokens=max_T)
    model = SynapseEndToEndModel(model_config).to(device)
    
    if config.compile_model:
        log.info("Optimizing model with torch.compile...")
        model = torch.compile(model)

    # Set normalization from training data
    model.normalized_lift.set_normalization(
        norm_stats.anchor_mu_tensor().to(device),
        norm_stats.anchor_sigma_tensor().to(device),
    )

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %d params (%d trainable)", param_count, trainable_count)

    # ── 5. Optimizer, scheduler, EMA ──────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    train_batches_per_epoch = min(len(train_loader), config.max_train_batches or len(train_loader))
    total_steps = config.epochs * train_batches_per_epoch
    warmup_steps = config.warmup_epochs * train_batches_per_epoch
    scheduler = create_cosine_warmup_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=config.min_lr,
    )
    ema = EMAModel(model, decay=config.ema_decay)

    loss_config = config.to_loss_config()

    # ── 6. Training loop ──────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history: List[Dict[str, float]] = []
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    log.info(
        "Starting training: %d epochs, %d batches/epoch (%d effective)",
        config.epochs,
        len(train_loader),
        train_batches_per_epoch,
    )

    for epoch in range(config.epochs):
        epoch_start = time.time()
        model.train()

        # Update sampler epoch for deterministic shuffling
        train_sampler.set_epoch(epoch)

        epoch_losses = []
        epoch_grad_norms = []
        batch_times_s = []
        recent_losses: deque[Dict[str, float]] = deque(maxlen=max(config.log_interval, 1))
        recent_grad_norms: deque[float] = deque(maxlen=max(config.log_interval, 1))

        # TQDM Custom Progress Bar
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch + 1}/{config.epochs}", 
            leave=False, 
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )

        for batch_idx, batch in enumerate(pbar):
            if config.max_train_batches is not None and batch_idx >= config.max_train_batches:
                break
            batch_start = time.perf_counter()
            # Skip empty batches (all samples failed)
            if not batch:
                continue

            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward with AMP
            amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=device.type, enabled=config.use_amp, dtype=amp_dtype):
                output = model.forward_train(
                    batch,
                    use_anchors=config.use_anchors,
                    use_topology=config.use_topology,
                )
                # Loss
                loss, loss_dict = combined_loss(output, batch, epoch, loss_config)

            # Backward with scaler
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Unscale before clipping
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), config.gradient_clip).item()
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # EMA update
            ema.update(model)

            epoch_losses.append(loss_dict)
            epoch_grad_norms.append(grad_norm)
            recent_losses.append(loss_dict)
            recent_grad_norms.append(grad_norm)
            batch_times_s.append(time.perf_counter() - batch_start)

            # Update progress bar postfix
            pbar.set_postfix({
                "loss": f"{loss_dict['loss_total']:.4f}",
                "act": f"{loss_dict['loss_action']:.4f}",
                "spr": f"{loss_dict['loss_sparsity']:.4f}",
                "topo": f"{loss_dict['loss_topo_reg']:.4f}",
                "grad": f"{grad_norm:.2f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

            if (batch_idx + 1) % config.log_interval == 0:
                mean_total = float(np.mean([entry["loss_total"] for entry in recent_losses]))
                mean_action = float(np.mean([entry["loss_action"] for entry in recent_losses]))
                mean_sparsity = float(np.mean([entry["loss_sparsity"] for entry in recent_losses]))
                mean_topo = float(np.mean([entry["loss_topo_reg"] for entry in recent_losses]))
                mean_grad = float(np.mean(recent_grad_norms))
                mean_batch_time = float(np.mean(batch_times_s[-len(recent_losses):])) if recent_losses else float("nan")
                throughput = config.batch_size / max(mean_batch_time, 1e-8)
                log.info(
                    "[Epoch %3d/%d | Batch %4d/%d] loss=%.5f | action=%.5f | sparsity=%.5f | topo=%.5f | grad=%.3f | lr=%.2e | %.2f samples/s",
                    epoch + 1,
                    config.epochs,
                    batch_idx + 1,
                    train_batches_per_epoch,
                    mean_total,
                    mean_action,
                    mean_sparsity,
                    mean_topo,
                    mean_grad,
                    optimizer.param_groups[0]["lr"],
                    throughput,
                )

        # Close the progress bar cleanly
        pbar.close()

        # Aggregate epoch stats
        if not epoch_losses:
            log.warning("Epoch %d: no valid batches!", epoch + 1)
            continue

        avg_loss = np.mean([d["loss_total"] for d in epoch_losses])
        avg_action = np.mean([d["loss_action"] for d in epoch_losses])
        avg_sparsity = np.mean([d["loss_sparsity"] for d in epoch_losses])
        avg_topo = np.mean([d["loss_topo_reg"] for d in epoch_losses])
        avg_grad = np.mean(epoch_grad_norms)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        avg_batch_time = float(np.mean(batch_times_s)) if batch_times_s else float("nan")

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(avg_loss),
            "train_action_loss": float(avg_action),
            "train_sparsity_loss": float(avg_sparsity),
            "train_topology_loss": float(avg_topo),
            "avg_grad_norm": float(avg_grad),
            "avg_batch_time_s": avg_batch_time,
            "lr": float(current_lr),
            "epoch_time_s": float(epoch_time),
        }

        # ── Validation ────────────────────────────────────────────
        if (epoch + 1) % config.val_check_interval == 0 or epoch == 0:
            val_metrics = evaluate_train_path(
                model,
                val_loader,
                loss_config,
                epoch,
                device,
                amp_enabled=config.use_amp,
                use_anchors=config.use_anchors,
                use_topology=config.use_topology,
                max_batches=config.max_eval_batches,
            )
            epoch_record.update(val_metrics)

            log.info(
                "[Epoch %3d/%d] train=%.5f | action=%.5f | sparsity=%.5f | topo=%.5f | val=%.5f | mse=%.5f | deploy_gap=%s | lr=%.2e | grad=%.3f | %.1fs",
                epoch + 1, config.epochs,
                avg_loss,
                avg_action,
                avg_sparsity,
                avg_topo,
                val_metrics.get("val_loss", float("nan")),
                val_metrics.get("val_action_mse", float("nan")),
                f"{val_metrics.get('gap_mse_relative', float('nan')):.5f}" if "gap_mse_relative" in val_metrics else "n/a",
                current_lr,
                avg_grad,
                epoch_time,
            )

            # Best model check
            val_loss = val_metrics.get("val_loss", float("inf"))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    artifacts["best_checkpoint"],
                    model, optimizer, scheduler, ema,
                    CheckpointMetadata(
                        epoch=epoch + 1,
                        metrics=val_metrics,
                        config=asdict(config),
                        seed=config.seed,
                    ),
                    normalization={
                        "mu": norm_stats.anchor_mu_tensor(),
                        "sigma": norm_stats.anchor_sigma_tensor(),
                    },
                )
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    log.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch + 1, config.early_stopping_patience,
                    )
                    break
        else:
            log.info(
                "[Epoch %3d/%d] train=%.5f | action=%.5f | sparsity=%.5f | topo=%.5f | lr=%.2e | grad=%.3f | %.1fs",
                epoch + 1, config.epochs, avg_loss, avg_action, avg_sparsity, avg_topo, current_lr, avg_grad, epoch_time,
            )

        # Periodic checkpoint
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch + 1:04d}.pt",
                model, optimizer, scheduler, ema,
                CheckpointMetadata(
                    epoch=epoch + 1,
                    metrics=epoch_record,
                    config=asdict(config),
                    seed=config.seed,
                ),
            )

        history.append(epoch_record)
        
        if config.use_wandb:
            import wandb
            wandb.log(epoch_record, step=epoch + 1)

    # ── 7. Final evaluation ───────────────────────────────────────
    log.info("=" * 70)
    log.info("Final evaluation on validation set")
    log.info("=" * 70)

    # Load best model
    best_ckpt = artifacts["best_checkpoint"]
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model)

    final_metrics = full_evaluation(
        model,
        val_loader,
        loss_config,
        epoch,
        device,
        amp_enabled=config.use_amp,
        use_anchors=config.use_anchors,
        use_topology=config.use_topology,
        max_batches=config.max_eval_batches,
    )

    log.info("Final results:")
    for k, v in sorted(final_metrics.items()):
        log.info("  %s: %.6f", k, v)
    if final_metrics.get("gap_mse_relative", 0.0) > 0.10:
        log.warning(
            "Train-deploy gap is above the 10%% target: %.4f",
            final_metrics["gap_mse_relative"],
        )
    if config.use_anchors and final_metrics.get("val_anchor_util", 0.0) < 0.05:
        log.warning(
            "Anchor utilization is extremely low (%.4f). Inspect selector behavior before large-scale runs.",
            final_metrics["val_anchor_util"],
        )
    if final_metrics.get("deploy_failed_batches", 0.0) > 0:
        log.warning(
            "Deploy evaluation encountered failed batches: %.0f",
            final_metrics["deploy_failed_batches"],
        )

    # Save training history
    history_path = artifacts["history"]
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    log.info("Training history saved to %s", history_path)

    # Save final metrics
    metrics_path = artifacts["final_metrics"]
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    # ── 8. Automated Analysis Pipeline ────────────────────────────
    log.info("=" * 70)
    log.info("Running automated analysis pipeline")
    log.info("=" * 70)

    if config.analysis_enabled:
        try:
            from ..analysis.anchor_analysis import (
                collect_anchor_placements,
                plot_anchor_heatmap,
                save_anchor_analysis,
            )
            from ..analysis.learning_curves import plot_learning_curves
            from ..analysis.topology_analysis import (
                collect_topology_features,
                plot_topology_tsne,
                save_topology_analysis,
            )

            analysis_dir = artifacts["analysis_dir"]
            analysis_dir.mkdir(parents=True, exist_ok=True)

            try:
                plot_learning_curves(history, analysis_dir, title_prefix=f"Z2 E2E (Seed {config.seed})")
                log.info("  [1/3] Learning curves generated.")
            except Exception as exc:
                log.error("Failed to generate learning curves: %s", exc)

            if config.use_anchors:
                try:
                    anchor_results = collect_anchor_placements(
                        model,
                        val_loader,
                        device,
                        max_batches=config.max_analysis_batches,
                    )
                    plot_anchor_heatmap(
                        anchor_results,
                        analysis_dir,
                        num_samples=config.anchor_heatmap_samples,
                        title=f"Anchor Placement (Seed {config.seed})",
                    )
                    anchor_stats = save_anchor_analysis(anchor_results, analysis_dir, K=config.K)
                    final_metrics.update({f"anchor/{k}": v for k, v in anchor_stats.items()})
                    log.info("  [2/3] Anchor analysis completed.")
                except Exception as exc:
                    log.error("Failed to run anchor analysis: %s", exc)
            else:
                log.info("  [2/3] Anchor analysis skipped (use_anchors=False).")

            if config.use_topology:
                try:
                    topo_results = collect_topology_features(
                        model,
                        val_loader,
                        device,
                        max_batches=config.max_analysis_batches,
                    )
                    plot_topology_tsne(
                        topo_results,
                        analysis_dir,
                        title=f"Topology t-SNE (Seed {config.seed})",
                        max_samples=config.topology_tsne_max_samples,
                        perplexity=config.topology_tsne_perplexity,
                        random_seed=config.topology_tsne_random_seed,
                    )
                    topo_stats = save_topology_analysis(topo_results, analysis_dir)
                    final_metrics.update({f"topology/{k}": v for k, v in topo_stats.items()})
                    log.info("  [3/3] Topology analysis completed.")
                except Exception as exc:
                    log.error("Failed to run topology analysis: %s", exc)
            else:
                log.info("  [3/3] Topology analysis skipped (use_topology=False).")

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(final_metrics, f, indent=2)

        except ImportError as exc:
            log.warning("Analysis scripts not available or failed to import: %s", exc)
    else:
        log.info("Analysis pipeline disabled by configuration.")

    manifest = {
        "seed": config.seed,
        "output_dir": str(run_dir),
        "train_lmdb": train_lmdb_path,
        "val_lmdb": val_lmdb_path,
        "best_checkpoint": str(artifacts["best_checkpoint"]),
        "final_metrics": str(metrics_path),
        "training_history": str(history_path),
        "analysis_dir": str(artifacts["analysis_dir"]),
        "resolved_config": str(artifacts["resolved_config"]),
        "best_val_loss": best_val_loss,
        "final_epoch": history[-1]["epoch"] if history else 0,
        "artifacts_relative": {
            "best_checkpoint": relativize_to_project(artifacts["best_checkpoint"]),
            "final_metrics": relativize_to_project(metrics_path),
            "training_history": relativize_to_project(history_path),
            "analysis_dir": relativize_to_project(artifacts["analysis_dir"]),
            "resolved_config": relativize_to_project(artifacts["resolved_config"]),
        },
    }
    save_json(artifacts["manifest"], manifest)
    log.info("Run manifest saved to %s", artifacts["manifest"])

    if config.use_wandb:
        import wandb
        wandb.log({"final/" + k: v for k, v in final_metrics.items()})
        wandb.finish()

    return {
        "final_metrics": final_metrics,
        "history": history,
        "best_val_loss": best_val_loss,
    }
