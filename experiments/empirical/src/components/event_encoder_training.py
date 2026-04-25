"""
EMP-01: Event Encoder Training Validation

Demonstrates that the learned EventEncoder (synapse_arch/event_encoder.py)
produces better event scores than the non-learned sharp encoder after
supervised training on synthetic data with known ground-truth events.

Spec: docs/implementation/phase2_empirical_validation/01_event_encoder.md
Z2 Reference: §3–4 of 02_rigorous_architecture.md
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Project imports
from synapse_core.event_encoder import sharp_event_score
from synapse_core.anchor_selector import solve_relaxed_selector, hard_projection
from synapse_arch.event_encoder import EventEncoder
from experiments.common.trajectory_generators import piecewise_constant
from experiments.empirical.common.metrics import match_f1
from experiments.empirical.common.seed_runner import run_multi_seed
from experiments.empirical.common.emp_config import load_emp_config, get_device_config, get_dataloader_kwargs
from experiments.empirical.common.data_saver import save_experiment_npz

log = logging.getLogger(__name__)


# ── Data Generation ────────────────────────────────────────────────────────

def generate_event_detection_dataset(
    n_samples: int, T: int, d: int, noise_std: float, seed: int,
) -> List[Dict[str, Any]]:
    """
    Generate samples of (trajectory, event_mask, change_points).
    event_mask[t] = 1.0 if t is a known change point, 0.0 otherwise.
    """
    rng = np.random.default_rng(seed)
    samples: List[Dict[str, Any]] = []
    for i in range(n_samples):
        num_events = int(rng.integers(3, 8))
        candidates = np.arange(10, T - 10)
        if len(candidates) < num_events:
            candidates = np.arange(2, T - 2)
        chosen = rng.choice(candidates, size=min(num_events, len(candidates)), replace=False)
        cps = sorted(chosen.tolist())

        traj, gt_cps = piecewise_constant(d, T, cps, jump_magnitude=5.0, seed=seed + i)
        traj = traj + rng.normal(scale=noise_std, size=traj.shape)

        event_mask = np.zeros(T, dtype=np.float32)
        for cp in gt_cps:
            event_mask[cp] = 1.0

        samples.append({
            "trajectory": traj.astype(np.float32),
            "event_mask": event_mask,
            "change_points": gt_cps,
        })
    return samples


# ── Custom Weighted Loss ───────────────────────────────────────────────────

class WeightedMSELoss(nn.Module):
    """MSE Loss with higher weight for positive (event) samples."""
    def __init__(self, pos_weight: float = 20.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = torch.where(target > 0.5, self.pos_weight, 1.0)
        return torch.mean(weight * (pred - target) ** 2)


# ── Lightning Module for Event Encoder ─────────────────────────────────────

class EventEncoderLitModule(pl.LightningModule):
    """Lightning wrapper for training EventEncoder with MSE on event masks."""

    def __init__(self, input_dim: int, hidden_dim: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = EventEncoder(input_dim, hidden_dim)
        self.loss_fn = WeightedMSELoss(pos_weight=20.0)
        self.lr = lr

    def forward(self, x: torch.Tensor):
        return self.encoder(x)

    def _shared_step(self, batch, stage: str):
        traj, mask = batch
        _, scores = self.encoder(traj)
        loss = self.loss_fn(scores, mask)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ── Lightning Module for Conv1D Baseline ───────────────────────────────────

class Conv1DDetector(nn.Module):
    """1D-CNN event detector baseline."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x.transpose(1, 2))  # [B,d,T] -> [B,1,T]
        return out.squeeze(1)


class Conv1DLitModule(pl.LightningModule):
    """Lightning wrapper for Conv1D detector."""

    def __init__(self, input_dim: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.detector = Conv1DDetector(input_dim)
        self.loss_fn = WeightedMSELoss(pos_weight=20.0)
        self.lr = lr

    def forward(self, x):
        return self.detector(x)

    def _shared_step(self, batch, stage):
        traj, mask = batch
        scores = self.detector(traj)
        loss = self.loss_fn(scores, mask)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ── Lightning Training Helper ──────────────────────────────────────────────

def _train_with_lightning(
    lit_model: pl.LightningModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int,
    patience: int,
    device_cfg: Dict[str, Any],
) -> pl.LightningModule:
    """Train a Lightning module with early stopping and best-checkpoint."""
    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val/loss", patience=patience, mode="min"),
        pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1),
        pl.callbacks.TQDMProgressBar(refresh_rate=10, leave=False),
    ]
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,  # no disk logging for sub-experiments
        **device_cfg,
    )
    trainer.fit(lit_model, train_loader, val_loader)
    # Load best checkpoint weights
    if trainer.checkpoint_callback.best_model_path:
        best = type(lit_model).load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
        )
        lit_model.load_state_dict(best.state_dict())
    return lit_model


# ── Evaluation ─────────────────────────────────────────────────────────────

def _evaluate_event_detection(
    score_fn,
    test_data: List[Dict[str, Any]],
    tolerance: int,
    r: int,
    lam: float,
) -> Dict[str, float]:
    """
    For each test sample:
      1. Get scores (learned or baseline)
      2. Apply relaxed selector + hard projection with K=len(gt_cps)
      3. Compute F1 between detected events and ground truth
    """
    f1_scores: List[float] = []
    for sample in test_data:
        scores = score_fn(sample)
        K = len(sample["change_points"])
        if K == 0:
            f1_scores.append(1.0)
            continue
        scores_f64 = scores.astype(np.float64)
        y_star = solve_relaxed_selector(scores_f64, K, r, lam, solver="osqp")
        detected = hard_projection(y_star, K, r)
        f1 = match_f1(detected, sample["change_points"], tolerance)
        f1_scores.append(f1)

    return {
        "mean_f1": float(np.mean(f1_scores)),
        "std_f1": float(np.std(f1_scores, ddof=1)) if len(f1_scores) > 1 else 0.0,
    }


# ── Score Functions ────────────────────────────────────────────────────────

def _make_learned_score_fn(lit_model: EventEncoderLitModule, device: torch.device):
    """Return callable extracting scores from a trained EventEncoder."""
    lit_model.eval()
    lit_model.to(device)

    def score_fn(sample: Dict[str, Any]) -> np.ndarray:
        traj = torch.from_numpy(sample["trajectory"]).unsqueeze(0).to(device)
        with torch.no_grad():
            _, scores_t = lit_model.encoder(traj)
        return scores_t.squeeze(0).cpu().numpy()
    return score_fn


def _make_cnn_score_fn(lit_model: Conv1DLitModule, device: torch.device):
    """Return callable extracting scores from a trained Conv1D detector."""
    lit_model.eval()
    lit_model.to(device)

    def score_fn(sample: Dict[str, Any]) -> np.ndarray:
        traj = torch.from_numpy(sample["trajectory"]).unsqueeze(0).to(device)
        with torch.no_grad():
            scores_t = torch.relu(lit_model.detector(traj))
        return scores_t.squeeze(0).cpu().numpy()
    return score_fn


def _sharp_score_fn(sample: Dict[str, Any]) -> np.ndarray:
    return sharp_event_score(sample["trajectory"].astype(np.float64)).astype(np.float32)


def _diff_norm_score_fn(sample: Dict[str, Any]) -> np.ndarray:
    traj = sample["trajectory"]
    scores = np.zeros(len(traj), dtype=np.float32)
    if len(traj) > 1:
        scores[1:] = np.linalg.norm(np.diff(traj, axis=0), axis=1).astype(np.float32)
    return scores


# ── Single-Seed Experiment ─────────────────────────────────────────────────

def run_single_seed(config: Any, seed: int) -> Dict[str, float]:
    """Run EMP-01 for a single seed."""
    pl.seed_everything(seed, workers=True)

    cfg = config
    T = cfg.trajectory.T
    d = cfg.trajectory.d
    n_train = cfg.data.n_train
    n_val = cfg.data.n_val
    n_test = cfg.data.n_test
    noise_std = cfg.data.noise_std
    epochs = cfg.training.full_epochs
    lr = cfg.training.lr
    bs = cfg.training.batch_size
    patience = getattr(cfg.training, "patience", 10)
    hidden_dim = cfg.model.hidden_dim if hasattr(cfg, "model") else 64
    tolerance = cfg.evaluation.tolerance if hasattr(cfg, "evaluation") else 2
    K_mem = cfg.memory.K
    r = cfg.memory.r
    lam = cfg.memory.lam

    device_cfg = get_device_config(cfg)
    dl_kwargs = get_dataloader_kwargs(cfg)

    # Generate data
    train_data = generate_event_detection_dataset(n_train, T, d, noise_std, seed)
    val_data = generate_event_detection_dataset(n_val, T, d, noise_std, seed + 10000)
    test_data = generate_event_detection_dataset(n_test, T, d, noise_std, seed + 20000)

    # Build DataLoaders
    def _to_loader(data, shuffle):
        trajs = torch.from_numpy(np.stack([s["trajectory"] for s in data]))
        masks = torch.from_numpy(np.stack([s["event_mask"] for s in data]))
        return DataLoader(
            TensorDataset(trajs, masks),
            batch_size=bs, shuffle=shuffle, **dl_kwargs,
        )

    train_loader = _to_loader(train_data, shuffle=True)
    val_loader = _to_loader(val_data, shuffle=False)

    # ── Train learned EventEncoder via Lightning ──
    encoder_lit = EventEncoderLitModule(input_dim=d, hidden_dim=hidden_dim, lr=lr)
    encoder_lit = _train_with_lightning(
        encoder_lit, train_loader, val_loader, epochs, patience, device_cfg,
    )

    # ── Train CNN baseline via Lightning ──
    cnn_lit = Conv1DLitModule(input_dim=d, lr=lr)
    cnn_lit = _train_with_lightning(
        cnn_lit, train_loader, val_loader, epochs, patience, device_cfg,
    )

    # ── Evaluate all methods (move to CPU for numpy-based evaluation) ──
    eval_device = torch.device("cpu")
    eval_kwargs = dict(tolerance=tolerance, r=r, lam=lam)

    learned_results = _evaluate_event_detection(
        _make_learned_score_fn(encoder_lit, eval_device), test_data, **eval_kwargs,
    )
    cnn_results = _evaluate_event_detection(
        _make_cnn_score_fn(cnn_lit, eval_device), test_data, **eval_kwargs,
    )
    sharp_results = _evaluate_event_detection(
        _sharp_score_fn, test_data, **eval_kwargs,
    )
    diff_results = _evaluate_event_detection(
        _diff_norm_score_fn, test_data, **eval_kwargs,
    )

    save_experiment_npz(
        experiment_id="EMP-01",
        seed=seed,
        data={
            "test_trajectories": np.stack([s["trajectory"] for s in test_data]),
            "test_masks": np.stack([s["event_mask"] for s in test_data]),
        },
        output_dir=cfg.output_dir
    )

    return {
        "learned_f1": learned_results["mean_f1"],
        "sharp_f1": sharp_results["mean_f1"],
        "cnn_f1": cnn_results["mean_f1"],
        "diff_f1": diff_results["mean_f1"],
    }


# ── Entry Point ────────────────────────────────────────────────────────────

def run_experiment(config: Any = None) -> Dict[str, Any]:
    """Run EMP-01 across all seeds and return aggregated report."""
    if config is None:
        config = load_emp_config("EMP-01")

    report = run_multi_seed(
        experiment_fn=run_single_seed,
        config=config,
        seeds=config.training.seeds,
        experiment_id="EMP-01",
        output_dir=config.output_dir,
    )

    agg = report["aggregated"]
    learned_mean = agg.get("learned_f1", {}).get("mean", 0.0)
    sharp_mean = agg.get("sharp_f1", {}).get("mean", 0.0)
    margin = 0.05
    if hasattr(config, "acceptance_gates"):
        margin = getattr(config.acceptance_gates, "learned_vs_sharp_margin", 0.05)
    passed = learned_mean > sharp_mean + margin

    report["experiment_name"] = "Event Encoder Training"
    report["pass_criterion"] = f"learned_f1 > sharp_f1 + {margin}"
    report["passed"] = passed

    log.info("[EMP-01] learned_f1=%.4f  sharp_f1=%.4f  passed=%s",
             learned_mean, sharp_mean, passed)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if str(Path(__file__).resolve().parent.parent.parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
    result = run_experiment()
    print(f"\nEMP-01 PASSED: {result['passed']}")
