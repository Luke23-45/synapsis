"""
EMP-03: Normalized Lift Training Validation

Demonstrates that the learned geometric lift (NormalizedLift) organizes the
anchor point cloud in a task-discriminative way after contrastive training,
and that normalization N(v) is essential for this organization.

Spec: docs/implementation/phase2_empirical_validation/03_lift_training.md
Z2 Reference: §8–9 of 02_rigorous_architecture.md
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from synapse_core.event_encoder import sharp_event_score
from synapse_core.anchor_selector import solve_relaxed_selector, hard_projection, build_anchors
from synapse_core.geometric_lift import anchor_vectors
from synapse_arch.normalized_lift import NormalizedLift
from experiments.empirical.common.tasks import generate_topology_dataset
from experiments.empirical.common.seed_runner import run_multi_seed
from experiments.empirical.common.emp_config import load_emp_config, get_device_config
from experiments.empirical.common.math_utils import ridge_probe_accuracy
from experiments.empirical.common.data_saver import save_experiment_npz

log = logging.getLogger(__name__)


# ── Anchor Extraction ──────────────────────────────────────────────────────

def _extract_anchors_numpy(
    trajectory: np.ndarray, K: int, r: int, lam: float,
) -> np.ndarray:
    """Run Z2 core operator up to anchor vector extraction. Returns V [m, d+3]."""
    traj_f64 = trajectory.astype(np.float64)
    scores = sharp_event_score(traj_f64)
    y_star = solve_relaxed_selector(scores, K, r, lam, solver="osqp")
    indices = hard_projection(y_star, K, r)
    if not indices:
        return np.zeros((0, traj_f64.shape[1] + 3), dtype=np.float64)
    anchors = build_anchors(indices, traj_f64, scores)
    return anchor_vectors(anchors)


# ── Contrastive Loss ───────────────────────────────────────────────────────

def _contrastive_loss(
    lifted_batch: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1,
) -> torch.Tensor:
    """NT-Xent contrastive loss on cloud centroids."""
    centroids = lifted_batch.mean(dim=1)
    centroids = F.normalize(centroids, dim=-1)
    B = centroids.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=centroids.device, requires_grad=True)
    sim = torch.mm(centroids, centroids.T) / temperature
    label_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    label_eq.fill_diagonal_(0)
    pos_count = label_eq.sum(dim=1)
    valid_mask = pos_count > 0
    if not valid_mask.any():
        return torch.tensor(0.0, device=centroids.device, requires_grad=True)
    exp_sim = torch.exp(sim)
    exp_sim.fill_diagonal_(0)
    pos_sum = (exp_sim * label_eq).sum(dim=1)
    all_sum = exp_sim.sum(dim=1)
    loss = -torch.log(pos_sum / all_sum.clamp_min(1e-8) + 1e-8)
    return loss[valid_mask].mean()


# ── Lightning Module for Contrastive Lift ──────────────────────────────────

class ContrastiveLiftLitModule(pl.LightningModule):
    """Lightning module for contrastive training of NormalizedLift W_Θ."""

    def __init__(
        self,
        input_dim: int,
        lift_dim: int,
        lr: float = 5e-3,
        temperature: float = 0.1,
        skip_normalization: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lift = NormalizedLift(input_dim=input_dim, lift_dim=lift_dim)
        self.lr = lr
        self.temperature = temperature
        if skip_normalization:
            self.lift.set_normalization(
                torch.zeros(input_dim), torch.ones(input_dim),
            )

    def forward(self, x):
        return self.lift(x)

    def _shared_step(self, batch, stage):
        vectors, labels = batch
        _, lifted = self.lift(vectors)
        loss = _contrastive_loss(lifted, labels, self.temperature)
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


# ── Data Preparation ───────────────────────────────────────────────────────

def _prepare_lift_dataset(
    cfg: Any, seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate topology-aware sequences, extract anchor vectors via Z2 core.
    Returns (vectors_padded [N, K, anchor_dim], labels [N]).
    """
    rng = np.random.default_rng(seed)
    n_samples = cfg.data.n_train + cfg.data.n_val + cfg.data.n_test
    samples = generate_topology_dataset(rng, n_samples, cfg.trajectory.T, cfg.trajectory.d, cfg.data.noise_std)

    anchor_dim = cfg.trajectory.d + 3
    K = cfg.memory.K
    all_vectors = np.zeros((len(samples), K, anchor_dim), dtype=np.float32)
    all_labels = np.zeros(len(samples), dtype=np.int64)

    for i, sample in enumerate(samples):
        V = _extract_anchors_numpy(sample.sequence, K, cfg.memory.r, cfg.memory.lam)
        m = min(len(V), K)
        if m > 0:
            all_vectors[i, :m, :] = V[:m].astype(np.float32)
        all_labels[i] = sample.label

    return all_vectors, all_labels


def _split(X, y, cfg, seed):
    """Split into train/val/test."""
    rng = np.random.default_rng(seed + 999)
    n = len(X)
    n_train = cfg.data.n_train
    n_val = cfg.data.n_val
    perm = rng.permutation(n)
    return (
        X[perm[:n_train]], y[perm[:n_train]],
        X[perm[n_train:n_train + n_val]], y[perm[n_train:n_train + n_val]],
        X[perm[n_train + n_val:]], y[perm[n_train + n_val:]],
    )


# ── Ridge Probe ────────────────────────────────────────────────────────────

def _ridge_probe_accuracy(X_train, y_train, X_test, y_test) -> float:
    """Linear probe via ridge regression — delegates to consolidated implementation."""
    return ridge_probe_accuracy(
        np.asarray(X_train, dtype=np.float64), y_train,
        np.asarray(X_test, dtype=np.float64), y_test,
    )


# ── Single-Seed Experiment ─────────────────────────────────────────────────

def run_single_seed(config: Any, seed: int) -> Dict[str, float]:
    """Run EMP-03 for a single seed."""
    pl.seed_everything(seed, workers=True)
    cfg = config

    anchor_dim = cfg.trajectory.d + 3
    lift_dim = cfg.model.lift_dim if hasattr(cfg, "model") else cfg.memory.k
    temperature = cfg.model.temperature if hasattr(cfg, "model") else 0.1
    epochs = cfg.training.full_epochs
    lr = cfg.training.lr
    bs = cfg.training.batch_size
    patience = getattr(cfg.training, "patience", 10)
    device_cfg = get_device_config(cfg)

    # Prepare data
    all_vectors, all_labels = _prepare_lift_dataset(cfg, seed)
    X_train, y_train, X_val, y_val, X_test, y_test = _split(all_vectors, all_labels, cfg, seed)

    results: Dict[str, float] = {}

    # ── LIFT-TRAINED: Contrastive training via Lightning ──
    dl_kwargs = {"num_workers": 0}
    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long(),
    ), batch_size=bs, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long(),
    ), batch_size=bs, shuffle=False, **dl_kwargs)

    # Compute normalization from training data
    flat_train = X_train.reshape(-1, anchor_dim)
    nonzero_mask = np.any(flat_train != 0, axis=1)

    pl.seed_everything(seed, workers=True)
    lit_trained = ContrastiveLiftLitModule(
        input_dim=anchor_dim, lift_dim=lift_dim, lr=lr, temperature=temperature,
    )
    if nonzero_mask.sum() > 1:
        mu_np = flat_train[nonzero_mask].mean(axis=0)
        sigma_np = flat_train[nonzero_mask].std(axis=0)
        sigma_np[sigma_np < 1e-6] = 1.0
        lit_trained.lift.set_normalization(
            torch.from_numpy(mu_np).float(), torch.from_numpy(sigma_np).float(),
        )

    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val/loss", patience=patience, mode="min"),
        pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1),
        pl.callbacks.TQDMProgressBar(refresh_rate=10, leave=False),
    ]
    trainer = pl.Trainer(
        max_epochs=epochs, callbacks=callbacks,
        enable_progress_bar=True, enable_model_summary=False, logger=False,
        **device_cfg,
    )
    trainer.fit(lit_trained, train_loader, val_loader)
    if trainer.checkpoint_callback.best_model_path:
        best = ContrastiveLiftLitModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
        )
        lit_trained.load_state_dict(best.state_dict())

    # Evaluate LIFT-TRAINED
    lit_trained.eval().cpu()
    with torch.no_grad():
        _, lifted_train = lit_trained.lift(torch.from_numpy(X_train).float())
        _, lifted_test = lit_trained.lift(torch.from_numpy(X_test).float())
    c_train = lifted_train.mean(dim=1).numpy()
    c_test = lifted_test.mean(dim=1).numpy()
    results["LIFT_TRAINED_probe_acc"] = _ridge_probe_accuracy(c_train, y_train, c_test, y_test)

    try:
        from sklearn.metrics import silhouette_score
        if len(np.unique(y_test)) > 1 and len(c_test) > len(np.unique(y_test)):
            results["LIFT_TRAINED_silhouette"] = float(silhouette_score(c_test, y_test))
        else:
            results["LIFT_TRAINED_silhouette"] = 0.0
    except ImportError:
        results["LIFT_TRAINED_silhouette"] = 0.0

    # ── LIFT-RANDOM: Random W_Θ ──
    pl.seed_everything(seed + 1000, workers=True)
    lift_random = NormalizedLift(input_dim=anchor_dim, lift_dim=lift_dim)
    lift_random.eval()
    with torch.no_grad():
        _, lr_train = lift_random(torch.from_numpy(X_train).float())
        _, lr_test = lift_random(torch.from_numpy(X_test).float())
    results["LIFT_RANDOM_probe_acc"] = _ridge_probe_accuracy(
        lr_train.mean(dim=1).numpy(), y_train, lr_test.mean(dim=1).numpy(), y_test,
    )
    try:
        from sklearn.metrics import silhouette_score
        lr_c = lr_test.mean(dim=1).numpy()
        if len(np.unique(y_test)) > 1 and len(lr_c) > len(np.unique(y_test)):
            results["LIFT_RANDOM_silhouette"] = float(silhouette_score(lr_c, y_test))
        else:
            results["LIFT_RANDOM_silhouette"] = 0.0
    except ImportError:
        results["LIFT_RANDOM_silhouette"] = 0.0

    # ── LIFT-PCA ──
    from numpy.linalg import svd
    flat_nz = flat_train[nonzero_mask] if nonzero_mask.sum() > 0 else flat_train
    mean_pca = flat_nz.mean(axis=0)
    _, _, Vt = svd(flat_nz - mean_pca, full_matrices=False)
    W_pca = Vt[:lift_dim, :]

    def _pca_centroids(X_all):
        flat = X_all.reshape(-1, anchor_dim)
        projected = (flat - mean_pca) @ W_pca.T
        return projected.reshape(X_all.shape[0], X_all.shape[1], lift_dim).mean(axis=1)

    results["LIFT_PCA_probe_acc"] = _ridge_probe_accuracy(
        _pca_centroids(X_train), y_train, _pca_centroids(X_test), y_test,
    )

    # ── LIFT-IDENTITY ──
    results["LIFT_IDENTITY_probe_acc"] = _ridge_probe_accuracy(
        X_train.mean(axis=1), y_train, X_test.mean(axis=1), y_test,
    )

    # ── LIFT-NO-NORM: trained W_Θ but bypass normalization ──
    pl.seed_everything(seed, workers=True)
    lit_no_norm = ContrastiveLiftLitModule(
        input_dim=anchor_dim, lift_dim=lift_dim, lr=lr, temperature=temperature,
        skip_normalization=True,
    )
    trainer2 = pl.Trainer(
        max_epochs=epochs, callbacks=[
            pl.callbacks.EarlyStopping(monitor="val/loss", patience=patience, mode="min"),
            pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1),
            pl.callbacks.TQDMProgressBar(refresh_rate=10, leave=False),
        ],
        enable_progress_bar=True, enable_model_summary=False, logger=False,
        **device_cfg,
    )
    trainer2.fit(lit_no_norm, train_loader, val_loader)
    if trainer2.checkpoint_callback.best_model_path:
        best_nn = ContrastiveLiftLitModule.load_from_checkpoint(
            trainer2.checkpoint_callback.best_model_path,
        )
        lit_no_norm.load_state_dict(best_nn.state_dict())

    lit_no_norm.eval().cpu()
    with torch.no_grad():
        _, nn_train = lit_no_norm.lift(torch.from_numpy(X_train).float())
        _, nn_test = lit_no_norm.lift(torch.from_numpy(X_test).float())
    results["LIFT_NO_NORM_probe_acc"] = _ridge_probe_accuracy(
        nn_train.mean(dim=1).numpy(), y_train, nn_test.mean(dim=1).numpy(), y_test,
    )

    save_experiment_npz("EMP-03", seed, {"X_test": X_test, "y_test": y_test, "X_train": X_train, "y_train": y_train}, cfg.output_dir)

    return results


# ── Entry Point ────────────────────────────────────────────────────────────

def run_experiment(config: Any = None) -> Dict[str, Any]:
    """Run EMP-03 across all seeds and return aggregated report."""
    if config is None:
        config = load_emp_config("EMP-03")

    report = run_multi_seed(
        experiment_fn=run_single_seed,
        config=config,
        seeds=config.training.seeds,
        experiment_id="EMP-03",
        output_dir=config.output_dir,
    )

    agg = report["aggregated"]
    trained_acc = agg.get("LIFT_TRAINED_probe_acc", {}).get("mean", 0.0)
    random_acc = agg.get("LIFT_RANDOM_probe_acc", {}).get("mean", 0.0)
    trained_sil = agg.get("LIFT_TRAINED_silhouette", {}).get("mean", 0.0)
    random_sil = agg.get("LIFT_RANDOM_silhouette", {}).get("mean", 0.0)
    passed = (trained_acc > random_acc) and (trained_sil > random_sil)

    report["experiment_name"] = "Normalized Lift Training"
    report["pass_criterion"] = "LIFT-TRAINED > LIFT-RANDOM on both probe_acc and silhouette"
    report["passed"] = passed

    log.info("[EMP-03] trained_acc=%.4f random_acc=%.4f  trained_sil=%.4f random_sil=%.4f  passed=%s",
             trained_acc, random_acc, trained_sil, random_sil, passed)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if str(Path(__file__).resolve().parent.parent.parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
    result = run_experiment()
    print(f"\nEMP-03 PASSED: {result['passed']}")
