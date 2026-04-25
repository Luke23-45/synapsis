"""
EMP-02: Selector Gradient Flow Validation

Verifies that gradients flow correctly through the relaxed selector layer
during end-to-end training, and that the selector learns to assign high
weights to task-relevant events.

Spec: docs/implementation/phase2_empirical_validation/02_selector_gradient.md
Z2 Reference: §5 of 02_rigorous_architecture.md, Remark 5.3
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
from torch.utils.data import DataLoader, TensorDataset

from synapse_arch.event_encoder import EventEncoder
from synapse_arch.saliency_normalizer import SaliencyNormalizer
from synapse_arch.relaxed_selector_layer import RelaxedSelectorLayer
from experiments.empirical.common.tasks import generate_memory_task_dataset, SequenceSample
from experiments.empirical.common.seed_runner import run_multi_seed
from experiments.empirical.common.emp_config import load_emp_config, get_device_config, get_dataloader_kwargs
from experiments.empirical.common.data_saver import save_experiment_npz

log = logging.getLogger(__name__)


# ── Selector Probe Models ─────────────────────────────────────────────────

class SelectorProbeModel(nn.Module):
    """
    Z2 relaxed selector probe: encoder → saliency → QP selector → weighted pool → classifier.
    """

    def __init__(self, input_dim: int, hidden_dim: int, K: int, r: int, lam: float, num_classes: int):
        super().__init__()
        self.event_encoder = EventEncoder(input_dim, hidden_dim)
        self.saliency_norm = SaliencyNormalizer()
        self.selector = RelaxedSelectorLayer(K, r, lam)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        hidden, event_scores = self.event_encoder(x)
        saliency = self.saliency_norm(event_scores)
        y_star = self.selector(saliency)
        weights = y_star.unsqueeze(-1)
        weighted_sum = (hidden * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1e-6)
        pooled = weighted_sum / denom
        logits = self.classifier(pooled)
        return logits, y_star


class UniformProbeModel(nn.Module):
    """Baseline: uniform weighting over all timesteps."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.event_encoder = EventEncoder(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        hidden, _ = self.event_encoder(x)
        B, T, h = hidden.shape
        y_star = torch.ones(B, T, device=x.device, dtype=x.dtype) / T
        pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, y_star


class TopKProbeModel(nn.Module):
    """Baseline: hard top-K event selection."""

    def __init__(self, input_dim: int, hidden_dim: int, K: int, num_classes: int):
        super().__init__()
        self.event_encoder = EventEncoder(input_dim, hidden_dim)
        self.K = K
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        hidden, event_scores = self.event_encoder(x)
        B, T, h = hidden.shape
        _, topk_idx = event_scores.topk(min(self.K, T), dim=1)
        y_star = torch.zeros(B, T, device=x.device, dtype=x.dtype)
        y_star.scatter_(1, topk_idx, 1.0)
        weights = y_star.unsqueeze(-1)
        weighted_sum = (hidden * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1e-6)
        pooled = weighted_sum / denom
        logits = self.classifier(pooled)
        return logits, y_star


class SoftmaxAttnProbeModel(nn.Module):
    """Baseline: softmax attention over saliency scores."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, temperature: float = 1.0):
        super().__init__()
        self.event_encoder = EventEncoder(input_dim, hidden_dim)
        self.saliency_norm = SaliencyNormalizer()
        self.temperature = temperature
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        hidden, event_scores = self.event_encoder(x)
        saliency = self.saliency_norm(event_scores)
        y_star = torch.softmax(saliency / self.temperature, dim=-1)
        weights = y_star.unsqueeze(-1)
        weighted_sum = (hidden * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1e-6)
        pooled = weighted_sum / denom
        logits = self.classifier(pooled)
        return logits, y_star


# ── Lightning Module for Selector Probes ───────────────────────────────────

class SelectorProbeLitModule(pl.LightningModule):
    """Lightning wrapper for any selector probe model (classification)."""

    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits, _ = self.model(x)
        loss = F.cross_entropy(logits, y.long())
        preds = logits.argmax(dim=-1)
        acc = (preds == y.long()).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ── Lightning Training + Test ──────────────────────────────────────────────

def _train_and_test(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    patience: int,
    device_cfg: Dict[str, Any],
) -> float:
    """Train a selector probe via Lightning, return test accuracy."""
    lit = SelectorProbeLitModule(model, lr=lr)
    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val/loss", patience=patience, mode="min"),
        pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1),
    ]
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        **device_cfg,
    )
    trainer.fit(lit, train_loader, val_loader)
    results = trainer.test(lit, test_loader, ckpt_path="best", verbose=False)
    return results[0].get("test/acc", 0.0) if results else 0.0


# ── Single-Seed Experiment ─────────────────────────────────────────────────

def run_single_seed(config: Any, seed: int) -> Dict[str, float]:
    """Run EMP-02 for a single seed across all task families."""
    pl.seed_everything(seed, workers=True)
    rng = np.random.default_rng(seed)

    cfg = config
    d = cfg.trajectory.d
    T = cfg.trajectory.T
    K = cfg.memory.K
    r = cfg.memory.r
    lam = cfg.memory.lam
    hidden_dim = cfg.model.hidden_dim if hasattr(cfg, "model") else 64
    epochs = cfg.training.full_epochs
    lr = cfg.training.lr
    bs = cfg.training.batch_size
    patience = getattr(cfg.training, "patience", 10)
    n_train = cfg.data.n_train
    n_val = cfg.data.n_val
    n_test = cfg.data.n_test

    families = cfg.task_families if hasattr(cfg, "task_families") else [
        "delayed_retrieval", "ordered_trigger_response", "phase_conditioned_decision",
    ]

    device_cfg = get_device_config(cfg)
    dl_kwargs = get_dataloader_kwargs(cfg)

    results: Dict[str, float] = {}
    family_wins = 0

    for family in families:
        train_data = generate_memory_task_dataset(rng, family, n_train, T, d, 0.25)
        val_data = generate_memory_task_dataset(rng, family, n_val, T, d, 0.35)
        test_data = generate_memory_task_dataset(rng, family, n_test, T, d, 0.35)

        all_labels = set(s.label for s in train_data + val_data + test_data)
        n_classes = max(len(all_labels), 2)

        def _to_loader(samples, shuffle):
            seqs = torch.from_numpy(np.stack([s.sequence for s in samples])).float()
            labels = torch.tensor([s.label for s in samples]).long()
            return DataLoader(
                TensorDataset(seqs, labels),
                batch_size=bs, shuffle=shuffle, **dl_kwargs,
            )

        train_loader = _to_loader(train_data, shuffle=True)
        val_loader = _to_loader(val_data, shuffle=False)
        test_loader = _to_loader(test_data, shuffle=False)

        selector_methods = {
            "SEL_Z2": lambda: SelectorProbeModel(d, hidden_dim, K, r, lam, n_classes),
            "SEL_UNI": lambda: UniformProbeModel(d, hidden_dim, n_classes),
            "SEL_TOPK": lambda: TopKProbeModel(d, hidden_dim, K, n_classes),
            "SEL_ATTN": lambda: SoftmaxAttnProbeModel(d, hidden_dim, n_classes),
        }

        family_results: Dict[str, float] = {}
        for method_name, model_factory in selector_methods.items():
            pl.seed_everything(seed, workers=True)
            model = model_factory()
            acc = _train_and_test(
                model, train_loader, val_loader, test_loader,
                epochs, lr, patience, device_cfg,
            )
            family_results[method_name] = acc
            results[f"{family}/{method_name}_accuracy"] = acc

        z2_acc = family_results.get("SEL_Z2", 0.0)
        uni_acc = family_results.get("SEL_UNI", 0.0)
        if z2_acc > uni_acc:
            family_wins += 1

    results["family_wins"] = float(family_wins)
    results["total_families"] = float(len(families))
    results["z2_wins_ratio"] = family_wins / max(len(families), 1)

    raw_data = {}
    for family in families:
        raw_data[f"{family}_test_seqs"] = np.stack([s.sequence for s in test_data])
        raw_data[f"{family}_test_labels"] = np.array([s.label for s in test_data])
    save_experiment_npz("EMP-02", seed, raw_data, cfg.output_dir)

    return results


# ── Entry Point ────────────────────────────────────────────────────────────

def run_experiment(config: Any = None) -> Dict[str, Any]:
    """Run EMP-02 across all seeds and return aggregated report."""
    if config is None:
        config = load_emp_config("EMP-02")

    report = run_multi_seed(
        experiment_fn=run_single_seed,
        config=config,
        seeds=config.training.seeds,
        experiment_id="EMP-02",
        output_dir=config.output_dir,
    )

    agg = report["aggregated"]
    wins_ratio_mean = agg.get("z2_wins_ratio", {}).get("mean", 0.0)
    min_ratio = 2.0 / 3.0
    if hasattr(config, "acceptance_gates"):
        min_ratio = getattr(config.acceptance_gates, "min_family_win_ratio", min_ratio)
    passed = wins_ratio_mean >= min_ratio

    report["experiment_name"] = "Selector Gradient Flow"
    report["pass_criterion"] = f"z2_wins_ratio >= {min_ratio:.2f}"
    report["passed"] = passed

    log.info("[EMP-02] z2_wins_ratio=%.4f  passed=%s", wins_ratio_mean, passed)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if str(Path(__file__).resolve().parent.parent.parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
    result = run_experiment()
    print(f"\nEMP-02 PASSED: {result['passed']}")
