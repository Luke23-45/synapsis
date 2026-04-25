"""
experiments/empirical/common/lightning_utils.py

Reusable PyTorch Lightning components for all EMP-* experiments.

Phase 2 Shared Infrastructure — see docs/implementation/phase2_empirical_validation/07_shared_infrastructure.md §2
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


class BaseExperimentModule(pl.LightningModule):
    """
    Shared Lightning module for all EMP experiments.

    Handles classification (CrossEntropy + Accuracy/F1) and
    regression (MSE) transparently through task_type selection.
    """

    def __init__(
        self,
        model: nn.Module,
        task_type: str = "classification",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        num_classes: int = 4,
    ):
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["model"])

        if task_type == "classification":
            self.loss_fn = nn.CrossEntropyLoss()
            # Use torchmetrics if available, otherwise manual tracking
            try:
                from torchmetrics import Accuracy, F1Score
                self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
                self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
                self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
                self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
                self._use_torchmetrics = True
            except ImportError:
                self._use_torchmetrics = False
        else:
            self.loss_fn = nn.MSELoss()
            self._use_torchmetrics = False

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self.model(x)

        if self.task_type == "classification":
            loss = self.loss_fn(logits, y.long())
            preds = logits.argmax(dim=-1)
            self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            # Always log accuracy — manual fallback if torchmetrics unavailable
            if self._use_torchmetrics:
                acc_metric = getattr(self, f"{stage}_acc")
                acc_metric(preds, y.long())
                self.log(f"{stage}/acc", acc_metric, prog_bar=True, on_step=False, on_epoch=True)
            else:
                acc = (preds == y.long()).float().mean()
                self.log(f"{stage}/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        else:
            target = y.float()
            if logits.dim() > 1 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            loss = self.loss_fn(logits, target)
            self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits = self._shared_step(batch, "test")
        if self.task_type == "classification" and self._use_torchmetrics:
            _, y = batch
            preds = logits.argmax(dim=-1)
            self.test_f1(preds, y.long())
            self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def make_data_module(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    batch_size: int = 32,
) -> Dict[str, DataLoader]:
    """Create train/val/test DataLoaders from numpy arrays."""
    def _to_loader(X, y, shuffle):
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        if np.issubdtype(np.asarray(y).dtype, np.integer):
            y_t = torch.from_numpy(np.asarray(y, dtype=np.int64))
        else:
            y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
        ds = TensorDataset(X_t, y_t)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=0, persistent_workers=False,
        )

    return {
        "train": _to_loader(X_train, y_train, shuffle=True),
        "val": _to_loader(X_val, y_val, shuffle=False),
        "test": _to_loader(X_test, y_test, shuffle=False),
    }


def quick_train_and_test(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    task_type: str = "classification",
    max_epochs: int = 50,
    lr: float = 1e-3,
    num_classes: int = 4,
    patience: int = 10,
    accelerator: str = "auto",
    devices: int = 1,
    precision: int = 32,
) -> Dict[str, float]:
    """
    Train a model with Lightning and return test metrics.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    loaders : dict
        Must contain "train", "val", "test" DataLoaders.
    task_type : str
        "classification" or "regression".
    accelerator : str
        Lightning accelerator ("auto", "gpu", "cpu").
    devices : int
        Number of devices to use.
    precision : int
        Training precision (32, 16, or "bf16-mixed").

    Returns
    -------
    Dict[str, float] with test metrics (e.g. "test/acc", "test/loss").
    """
    lit_model = BaseExperimentModule(
        model=model, task_type=task_type, lr=lr, num_classes=num_classes,
    )

    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val/loss", patience=patience, mode="min",
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="val/loss", mode="min", save_top_k=1,
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    trainer.fit(lit_model, loaders["train"], loaders["val"])
    results = trainer.test(lit_model, loaders["test"], ckpt_path="best", verbose=False)

    metrics: Dict[str, float] = {}
    if results:
        metrics = dict(results[0])
    return metrics
