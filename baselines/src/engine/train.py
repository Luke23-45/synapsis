"""
Training Engine — M1 Experiment
================================

Training loop for all conditions (A1, A2, B, B-Anchors, B-Topo).
Identical hyperparameters across conditions — the controlled variable.

Key decisions:
    - Optimizer: AdamW (weight_decay=0.01, betas=(0.9, 0.999))
    - LR schedule: Linear warmup (500 steps) + cosine decay
    - Early stopping: Patience 10 epochs on validation MSE
    - Gradient clipping: max_norm=1.0
    - Mixed precision: Optional AMP (if CUDA available)
    - Logging: Per-epoch train/val metrics
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.core.config import ExperimentConfig, Condition
from src.planner.planner_synapse import create_planner
from src.planner.planner import RoboticsPlannerBase
from src.data.dataset import DataLoader

# Import EMA and auxiliary regularizers from our robust Phase 3 codebase
from experiments.end_to_end.training.ema import EMAModel
from experiments.end_to_end.losses.auxiliary_losses import (
    sparsity_loss,
    topology_reg_loss,
)
from experiments.end_to_end.losses.combined_loss import LossConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def _cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Create a linear warmup + cosine decay LR schedule."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item()))

    return LambdaLR(optimizer, lr_lambda)


def _aux_weight_schedule(
    epoch: int,
    target_weight: float,
    ramp_start: int,
    ramp_end: int,
) -> float:
    if epoch < ramp_start:
        return 0.0
    if epoch >= ramp_end:
        return target_weight
    progress = (epoch - ramp_start) / max(ramp_end - ramp_start, 1)
    return target_weight * progress


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

@dataclass
class TrainState:
    """Mutable training state for a single condition run."""

    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    phase_accuracies: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Training engine for a single experimental condition.

    Parameters
    ----------
    config : ExperimentConfig
    train_loader : DataLoader
    val_loader : DataLoader
    output_dir : Path
        Directory for checkpoints and logs.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._best_model_state: Optional[Dict[str, torch.Tensor]] = None

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        log.info("Training device: %s", self.device)

        # Model
        self.model = create_planner(config).to(self.device)
        if config.training.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        log.info(
            "Created planner for condition=%s, params=%d",
            config.condition.value,
            self.model.num_trainable_params,
        )
        self._initialize_synapse_normalization()

        # Optimizer
        optimizer_kwargs = {
            "lr": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "betas": (config.training.beta1, config.training.beta2),
        }
        if (
            config.training.fused_adamw
            and self.device.type == "cuda"
            and "fused" in AdamW.__init__.__code__.co_varnames
        ):
            optimizer_kwargs["fused"] = True
        self.optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)

        # LR schedule
        total_steps = len(train_loader) * config.training.max_epochs
        self.scheduler = _cosine_warmup_scheduler(
            self.optimizer,
            warmup_steps=config.training.warmup_steps,
            total_steps=total_steps,
        )

        # EMA Setup
        self.ema = EMAModel(self.model, decay=0.999)

        # Modern AMP API (PyTorch 2.1+)
        self.use_amp = config.training.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.amp_dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

        # Training state
        self.state = TrainState()

        # Auxiliary-loss schedule is part of the experiment config, not a
        # hidden trainer default, so logs and optimization stay reproducible.
        self.loss_config = LossConfig(
            action_weight=config.training.action_loss_weight,
            sparsity_weight=config.training.sparsity_weight,
            topology_reg_weight=config.training.topology_reg_weight,
            action_beta=config.training.action_beta,
            aux_ramp_start=config.training.aux_ramp_start,
            aux_ramp_end=config.training.aux_ramp_end,
        )

    def _initialize_synapse_normalization(self) -> None:
        if not self.config.condition.uses_synapse:
            return
        architecture = getattr(self.model, "architecture", None)
        normalized_lift = getattr(architecture, "normalized_lift", None)
        if normalized_lift is None:
            return

        state_dim = self.config.structured_state_dim
        mu = torch.zeros(state_dim + 3, device=self.device, dtype=torch.float32)
        sigma = torch.ones(state_dim + 3, device=self.device, dtype=torch.float32)
        mu[0] = 0.5
        sigma[0] = 0.29
        mu[state_dim + 1] = 1.0
        sigma[state_dim + 1] = 1.0
        normalized_lift.set_normalization(mu, sigma)

    def _snapshot_ema_model(self) -> Dict[str, torch.Tensor]:
        live_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        self.ema.apply_to(self.model)
        ema_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        self.model.load_state_dict(live_state)
        return ema_state

    def _synapse_loss(
        self,
        pred_outputs,
        target_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        action_mse = F.mse_loss(pred_outputs.pred_actions, target_actions)
        alpha_sparsity = _aux_weight_schedule(
            self.state.epoch,
            self.loss_config.sparsity_weight,
            self.loss_config.aux_ramp_start,
            self.loss_config.aux_ramp_end,
        )
        alpha_topo = _aux_weight_schedule(
            self.state.epoch,
            self.loss_config.topology_reg_weight,
            self.loss_config.aux_ramp_start,
            self.loss_config.aux_ramp_end,
        )
        s_loss = sparsity_loss(pred_outputs.y_star)
        t_loss = topology_reg_loss(pred_outputs.topology_token)
        weighted_s_loss = s_loss * alpha_sparsity
        weighted_t_loss = t_loss * alpha_topo

        loss_dict = {
            "action_mse": action_mse,
            "sparsity_loss": s_loss,
            "topo_loss": t_loss,
            "weighted_sparsity_loss": weighted_s_loss,
            "weighted_topo_loss": weighted_t_loss,
            "alpha_sparsity": torch.tensor(alpha_sparsity, device=action_mse.device),
            "alpha_topo": torch.tensor(alpha_topo, device=action_mse.device),
        }
        total_loss = action_mse + weighted_s_loss + weighted_t_loss

        return total_loss, loss_dict

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch.

        Returns
        -------
        metrics : dict with keys: train_loss, action_mse, phase_acc, lr
        """
        self.model.train()
        total_loss = 0.0
        total_action_mse = 0.0
        total_sparsity_loss = 0.0
        total_topo_loss = 0.0
        total_weighted_sparsity_loss = 0.0
        total_weighted_topo_loss = 0.0
        total_phase_correct = 0
        total_phase_total = 0
        n_batches = 0

        batch_times_s = []
        samples_seen = 0

        # TQDM Custom Progress Bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.state.epoch + 1}/{self.config.training.max_epochs}",
            leave=False,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )

        for batch_idx, batch in enumerate(pbar):
            batch_start = time.perf_counter()
            # Non-blocking async transfer
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                if hasattr(self.model, "forward_train") and self.config.condition.uses_synapse:
                    pred_outputs = self.model.forward_train(batch)
                    pred_actions = pred_outputs.pred_actions
                    total_loss_val, loss_dict = self._synapse_loss(
                        pred_outputs,
                        batch["action_chunk"],
                    )
                    action_loss = loss_dict["action_mse"]
                    sparsity_val = loss_dict["sparsity_loss"]
                    topo_val = loss_dict["topo_loss"]
                    weighted_sparsity_val = loss_dict["weighted_sparsity_loss"]
                    weighted_topo_val = loss_dict["weighted_topo_loss"]
                    alpha_sparsity = float(loss_dict["alpha_sparsity"].item())
                    alpha_topo = float(loss_dict["alpha_topo"].item())
                else:
                    pred_actions = self.model(batch)
                    action_loss = F.mse_loss(pred_actions, batch["action_chunk"])
                    total_loss_val = self.loss_config.action_weight * action_loss
                    sparsity_val = torch.tensor(0.0, device=self.device)
                    topo_val = torch.tensor(0.0, device=self.device)
                    weighted_sparsity_val = torch.tensor(0.0, device=self.device)
                    weighted_topo_val = torch.tensor(0.0, device=self.device)
                    alpha_sparsity = 0.0
                    alpha_topo = 0.0

            self.scaler.scale(total_loss_val).backward()
            
            # Unscale before clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.gradient_clip_norm,
            ).item()
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()
            
            # Update EMA shadow weights
            self.ema.update(self.model)
            
            self.state.global_step += 1

            total_loss += total_loss_val.item()
            total_action_mse += action_loss.item()
            total_sparsity_loss += sparsity_val.item()
            total_topo_loss += topo_val.item()
            total_weighted_sparsity_loss += weighted_sparsity_val.item()
            total_weighted_topo_loss += weighted_topo_val.item()
            n_batches += 1
            batch_times_s.append(time.perf_counter() - batch_start)
            samples_seen += batch["proprio"].shape[0]

            # Update progress bar postfix
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(self.train_loader):
                postfix = {
                    "loss": f"{total_loss_val.item():.4f}",
                    "mse": f"{action_loss.item():.4f}",
                }
                if self.config.condition.uses_synapse:
                    postfix["sparse"] = f"{sparsity_val.item():.4f}"
                    postfix["topo"] = f"{topo_val.item():.4f}"
                    postfix["a_s"] = f"{alpha_sparsity:.3f}"
                    postfix["a_t"] = f"{alpha_topo:.3f}"
                postfix["grad"] = f"{grad_norm:.2f}"
                postfix["lr"] = f"{self.optimizer.param_groups[0]['lr']:.2e}"
                pbar.set_postfix(postfix)

        pbar.close()

        avg_loss = total_loss / max(1, n_batches)
        avg_mse = total_action_mse / max(1, n_batches)
        avg_sparsity = total_sparsity_loss / max(1, n_batches)
        avg_topo = total_topo_loss / max(1, n_batches)
        avg_weighted_sparsity = total_weighted_sparsity_loss / max(1, n_batches)
        avg_weighted_topo = total_weighted_topo_loss / max(1, n_batches)
        current_lr = self.optimizer.param_groups[0]["lr"]
        mean_batch_time = sum(batch_times_s) / max(1, len(batch_times_s))
        samples_per_second = samples_seen / max(1e-6, sum(batch_times_s))

        self.state.train_losses.append(avg_loss)

        return {
            "train_loss": avg_loss,
            "action_mse": avg_mse,
            "sparsity_loss": avg_sparsity,
            "topo_loss": avg_topo,
            "weighted_sparsity_loss": avg_weighted_sparsity,
            "weighted_topo_loss": avg_weighted_topo,
            "alpha_sparsity": _aux_weight_schedule(
                self.state.epoch,
                self.loss_config.sparsity_weight,
                self.loss_config.aux_ramp_start,
                self.loss_config.aux_ramp_end,
            ),
            "alpha_topo": _aux_weight_schedule(
                self.state.epoch,
                self.loss_config.topology_reg_weight,
                self.loss_config.aux_ramp_start,
                self.loss_config.aux_ramp_end,
            ),
            "lr": current_lr,
            "samples_per_second": samples_per_second,
            "mean_batch_time": mean_batch_time,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation.

        Returns
        -------
        metrics : dict with keys: val_loss, val_action_mse
        """
        # Evaluate with EMA weights, then restore the live training weights.
        live_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        self.ema.apply_to(self.model)
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        n_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            if hasattr(self.model, "forward_deploy") and self.config.condition.uses_synapse:
                pred_actions = self.model.forward_deploy(batch).pred_actions
            else:
                pred_actions = self.model(batch)
            action_loss = F.mse_loss(pred_actions, batch["action_chunk"])
            total_loss += action_loss.item()
            total_mse += action_loss.item()
            n_batches += 1

        # Restore normal weights for next training epoch
        self.model.load_state_dict(live_state)

        avg_loss = total_loss / max(1, n_batches)
        avg_mse = total_mse / max(1, n_batches)

        self.state.val_losses.append(avg_mse)

        return {
            "val_loss": avg_loss,
            "val_action_mse": avg_mse,
        }

    def train(self) -> TrainState:
        """Run the full training loop with early stopping.

        Returns
        -------
        TrainState
        """
        start_time = time.time()
        log.info(
            "Starting training: condition=%s, max_epochs=%d, patience=%d",
            self.config.condition.value,
            self.config.training.max_epochs,
            self.config.training.early_stopping_patience,
        )

        for epoch in range(self.config.training.max_epochs):
            self.state.epoch = epoch

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            val_mse = val_metrics["val_action_mse"]

            log_msg = (
                f"Epoch {epoch:3d} | train_total={train_metrics['train_loss']:.6f} | "
                f"train_mse={train_metrics['action_mse']:.6f} | val_mse={val_mse:.6f}"
            )
            if self.config.condition.uses_synapse:
                log_msg += (
                    f" | train_sparse={train_metrics['sparsity_loss']:.6f} "
                    f"| train_topo={train_metrics['topo_loss']:.6f}"
                    f" | alpha_sparse={train_metrics['alpha_sparsity']:.6f}"
                    f" | alpha_topo={train_metrics['alpha_topo']:.6f}"
                    f" | sparse_contrib={train_metrics['weighted_sparsity_loss']:.6f}"
                    f" | topo_contrib={train_metrics['weighted_topo_loss']:.6f}"
                )
            log_msg += f" | lr={train_metrics['lr']:.2e} | {train_metrics['samples_per_second']:.1f} samples/s"
            
            log.info(log_msg)

            # Early stopping check
            if val_mse < self.state.best_val_loss:
                self.state.best_val_loss = val_mse
                self.state.patience_counter = 0
                self._best_model_state = self._snapshot_ema_model()
                if self.config.training.save_checkpoints:
                    self.save_checkpoint("best.pt")
                    log.info("  → New best val_mse=%.6f, checkpoint saved", val_mse)
                else:
                    log.info("  → New best val_mse=%.6f (checkpoint saving disabled)", val_mse)
            else:
                self.state.patience_counter += 1
                if self.state.patience_counter >= self.config.training.early_stopping_patience:
                    log.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch,
                        self.state.patience_counter,
                    )
                    break

        self.state.elapsed_seconds = time.time() - start_time
        best_path = self.output_dir / "best.pt"
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)
        elif best_path.exists() and self.config.training.save_checkpoints:
            self.load_checkpoint("best.pt")
        log.info(
            "Training complete: %d epochs, %.1fs, best_val_mse=%.6f",
            self.state.epoch + 1,
            self.state.elapsed_seconds,
            self.state.best_val_loss,
        )

        return self.state

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "ema_state_dict": self.ema.state_dict(),
                "train_state": self.state,
                "config": self.config.to_dict(),
            },
            path,
        )

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        path = self.output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
        log.info("Loaded checkpoint from %s", path)
