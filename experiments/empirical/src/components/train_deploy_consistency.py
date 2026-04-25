"""
EMP-05: Training-Deploy Consistency Validation

Verifies that the relaxed training path (model.forward_train()) and the exact
deployment path (model.forward_deploy()) produce consistent outputs — i.e.,
a model trained via the relaxed path performs well when evaluated via the
exact path.

Spec: docs/implementation/phase2_empirical_validation/05_train_deploy.md
Z2 Reference: §13 of 02_rigorous_architecture.md
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from synapse_arch.model import SynapseEndToEndModel, SynapseArchitectureConfig
from experiments.empirical.common.tasks import generate_control_dataset
from experiments.empirical.common.seed_runner import run_multi_seed
from experiments.empirical.common.emp_config import load_emp_config, get_device_config, get_dataloader_kwargs
from experiments.empirical.common.data_saver import save_experiment_npz

log = logging.getLogger(__name__)


# ── Dataset ────────────────────────────────────────────────────────────────

class ControlEpisodeDataset(Dataset):
    """Wraps control episodes into (history, state, target_actions) tuples."""

    def __init__(self, episodes, chunk_size, max_T, input_dim, action_dim):
        self.samples: List[Dict[str, np.ndarray]] = []
        for ep in episodes:
            states = ep["states"]
            actions = ep["actions"]
            T_ep = len(states)
            for t in range(chunk_size, T_ep):
                end_t = min(t + chunk_size, T_ep)
                target = actions[t:end_t]
                if len(target) < chunk_size:
                    pad = np.zeros((chunk_size - len(target), action_dim), dtype=np.float32)
                    target = np.concatenate([target, pad], axis=0)
                history = states[:t]
                if len(history) < max_T:
                    pad = np.zeros((max_T - len(history), input_dim), dtype=np.float32)
                    history = np.concatenate([history, pad], axis=0)
                else:
                    history = history[-max_T:]
                self.samples.append({
                    "structured_history": history.astype(np.float32),
                    "structured_state": states[t].astype(np.float32),
                    "target_actions": target.astype(np.float32),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["structured_history"]),
            torch.from_numpy(s["structured_state"]),
            torch.from_numpy(s["target_actions"]),
        )


def _collate_fn(batch):
    histories, states, targets = zip(*batch)
    return (
        {"structured_history": torch.stack(histories), "structured_state": torch.stack(states)},
        torch.stack(targets),
    )


# ── Lightning Module for End-to-End Training ───────────────────────────────

class SynapseLitModule(pl.LightningModule):
    """
    Lightning wrapper for SynapseEndToEndModel training via forward_train.
    Handles GPU placement, gradient clipping, and LR scheduling.
    """

    def __init__(self, arch_config: SynapseArchitectureConfig, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["arch_config"])
        self.model = SynapseEndToEndModel(arch_config)
        self.lr = lr
        self.loss_fn = nn.MSELoss()

    def forward(self, batch_dict):
        return self.model.forward_train(batch_dict)

    def training_step(self, batch, batch_idx):
        batch_dict, targets = batch
        output = self.model.forward_train(batch_dict)
        loss = self.loss_fn(output.pred_actions, targets)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict, targets = batch
        output = self.model.forward_train(batch_dict)
        loss = self.loss_fn(output.pred_actions, targets)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ── Evaluation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate_path(model: SynapseEndToEndModel, loader: DataLoader, mode: str) -> float:
    """Evaluate using 'train' or 'deploy' forward path on CPU (deploy uses numpy)."""
    model.eval().cpu()
    mse_values: List[float] = []
    for batch_dict, targets in loader:
        # Move everything to CPU for deploy path compatibility
        batch_dict = {k: v.cpu() for k, v in batch_dict.items()}
        targets = targets.cpu()
        if mode == "train":
            output = model.forward_train(batch_dict)
        else:
            output = model.forward_deploy(batch_dict)
        mse = F.mse_loss(output.pred_actions, targets).item()
        mse_values.append(mse)
    return float(np.mean(mse_values)) if mse_values else float("inf")


@torch.no_grad()
def _measure_consistency(model: SynapseEndToEndModel, loader: DataLoader) -> Dict[str, float]:
    """Compare training-path and deployment-path outputs."""
    model.eval().cpu()
    overlaps: List[float] = []
    action_corrs: List[float] = []

    for batch_dict, _ in loader:
        batch_dict = {k: v.cpu() for k, v in batch_dict.items()}
        train_out = model.forward_train(batch_dict)
        deploy_out = model.forward_deploy(batch_dict)
        B = batch_dict["structured_history"].shape[0]
        for b in range(B):
            y_star = train_out.y_star[b].numpy()
            topk_train = set(np.argsort(y_star)[-model.config.K:])
            if b < len(deploy_out.exact_memory_states):
                deploy_indices = set(deploy_out.exact_memory_states[b].anchor_indices)
                if topk_train and deploy_indices:
                    overlaps.append(len(topk_train & deploy_indices) / model.config.K)
            a_train = train_out.pred_actions[b].numpy().flatten()
            a_deploy = deploy_out.pred_actions[b].numpy().flatten()
            if a_train.std() > 1e-8 and a_deploy.std() > 1e-8:
                corr = float(np.corrcoef(a_train, a_deploy)[0, 1])
                if np.isfinite(corr):
                    action_corrs.append(corr)

    return {
        "anchor_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
        "action_correlation_mean": float(np.mean(action_corrs)) if action_corrs else 0.0,
    }


# ── Single-Seed Experiment ─────────────────────────────────────────────────

def run_single_seed(config: Any, seed: int) -> Dict[str, float]:
    """Run EMP-05 for a single seed."""
    pl.seed_everything(seed, workers=True)
    cfg = config
    rng = np.random.default_rng(seed)

    arch = cfg.architecture if hasattr(cfg, "architecture") else cfg
    input_dim = getattr(arch, "input_dim", 8)
    action_dim = getattr(arch, "action_dim", 4)
    chunk_size = getattr(arch, "chunk_size", 5)
    hidden_dim = getattr(arch, "hidden_dim", 64)
    d_model = getattr(arch, "d_model", 128)
    num_heads = getattr(arch, "num_heads", 4)
    num_layers = getattr(arch, "num_layers", 2)
    ffn_ratio = getattr(arch, "ffn_ratio", 4)
    dropout = getattr(arch, "dropout", 0.1)
    K = cfg.memory.K
    r = cfg.memory.r
    lam = cfg.memory.lam
    k = cfg.memory.k
    Q = cfg.memory.Q
    T = cfg.trajectory.T
    n_episodes = getattr(cfg.data, "n_episodes", 100)
    epochs = cfg.training.full_epochs
    lr = cfg.training.lr
    bs = cfg.training.batch_size
    patience = getattr(cfg.training, "patience", 8)

    device_cfg = get_device_config(cfg)
    dl_kwargs = get_dataloader_kwargs(cfg)

    # Generate data
    episodes = generate_control_dataset(rng, n_episodes, T, input_dim, action_dim)
    n = len(episodes)
    perm = rng.permutation(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    train_eps = [episodes[i] for i in perm[:n_train]]
    val_eps = [episodes[i] for i in perm[n_train:n_train + n_val]]
    test_eps = [episodes[i] for i in perm[n_train + n_val:]]

    max_T = T
    train_ds = ControlEpisodeDataset(train_eps, chunk_size, max_T, input_dim, action_dim)
    val_ds = ControlEpisodeDataset(val_eps, chunk_size, max_T, input_dim, action_dim)
    test_ds = ControlEpisodeDataset(test_eps, chunk_size, max_T, input_dim, action_dim)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=_collate_fn, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=_collate_fn, **dl_kwargs)
    test_loader = DataLoader(test_ds, batch_size=min(bs, 8), shuffle=False, collate_fn=_collate_fn, num_workers=0)

    # Build model via Lightning
    arch_config = SynapseArchitectureConfig(
        input_dim=input_dim, action_dim=action_dim,
        action_chunk_size=chunk_size, hidden_dim=hidden_dim,
        d_model=d_model, num_heads=num_heads, num_layers=num_layers,
        ffn_ratio=ffn_ratio, dropout=dropout,
        K=K, r=r, lam=lam, Q=Q, k=k, max_history_tokens=K,
    )

    lit_model = SynapseLitModule(arch_config, lr=lr)
    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val/loss", patience=patience, mode="min"),
        pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1),
    ]

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        **device_cfg,
    )
    trainer.fit(lit_model, train_loader, val_loader)

    # Load best checkpoint
    if trainer.checkpoint_callback.best_model_path:
        best = SynapseLitModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            arch_config=arch_config,
        )
        lit_model.load_state_dict(best.state_dict())

    # Evaluate both paths (on CPU for deploy compatibility)
    model = lit_model.model
    train_path_mse = _evaluate_path(model, test_loader, mode="train")
    deploy_path_mse = _evaluate_path(model, test_loader, mode="deploy")

    gap_absolute = abs(deploy_path_mse - train_path_mse)
    gap_relative = gap_absolute / max(train_path_mse, 1e-8)
    consistency = _measure_consistency(model, test_loader)

    results = {
        "train_path_mse": train_path_mse,
        "deploy_path_mse": deploy_path_mse,
        "gap_absolute": gap_absolute,
        "gap_relative": gap_relative,
        "anchor_overlap": consistency["anchor_overlap_mean"],
        "action_correlation": consistency["action_correlation_mean"],
    }

    save_experiment_npz("EMP-05", seed, {
        "test_states": np.concatenate([ep["states"] for ep in test_eps]),
        "test_actions": np.concatenate([ep["actions"] for ep in test_eps]),
    }, cfg.output_dir)

    log.info("[EMP-05] seed=%d  train_mse=%.4f  deploy_mse=%.4f  gap_rel=%.4f  overlap=%.4f  corr=%.4f",
             seed, train_path_mse, deploy_path_mse, gap_relative,
             consistency["anchor_overlap_mean"], consistency["action_correlation_mean"])
    return results


# ── Entry Point ────────────────────────────────────────────────────────────

def run_experiment(config: Any = None) -> Dict[str, Any]:
    """Run EMP-05 across all seeds."""
    if config is None:
        config = load_emp_config("EMP-05")

    report = run_multi_seed(
        experiment_fn=run_single_seed,
        config=config,
        seeds=config.training.seeds,
        experiment_id="EMP-05",
        output_dir=config.output_dir,
    )

    agg = report["aggregated"]
    gap_rel = agg.get("gap_relative", {}).get("mean", 1.0)
    action_corr = agg.get("action_correlation", {}).get("mean", 0.0)

    max_gap = 0.25
    min_corr = 0.80
    if hasattr(config, "acceptance_gates"):
        max_gap = getattr(config.acceptance_gates, "max_gap_relative", max_gap)
        min_corr = getattr(config.acceptance_gates, "min_action_correlation", min_corr)

    passed = (gap_rel < max_gap) and (action_corr > min_corr)

    report["experiment_name"] = "Train-Deploy Consistency"
    report["pass_criterion"] = f"gap_relative < {max_gap} AND action_correlation > {min_corr}"
    report["passed"] = passed

    log.info("[EMP-05] gap_relative=%.4f  action_corr=%.4f  passed=%s", gap_rel, action_corr, passed)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if str(Path(__file__).resolve().parent.parent.parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
    result = run_experiment()
    print(f"\nEMP-05 PASSED: {result['passed']}")
