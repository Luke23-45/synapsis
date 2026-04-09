"""
Training and evaluation helpers for SYNAPSE empirical experiments (secondary/tertiary layers).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class FitResult:
    best_metric: float
    train_time: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loaders(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    task_type: str,
    lr: float,
    device: str,
) -> FitResult:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if task_type == "classification":
        criterion = nn.CrossEntropyLoss()
        eval_mode = "max"
    else:
        criterion = nn.MSELoss()
        eval_mode = "min"
    best_metric = -math.inf if eval_mode == "max" else math.inf
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb if task_type == "regression" else yb.long())
            loss.backward()
            optimizer.step()
        metric = evaluate_model(model, val_loader, task_type, device)["primary"]
        if (eval_mode == "max" and metric > best_metric) or (eval_mode == "min" and metric < best_metric):
            best_metric = metric
    train_time = time.perf_counter() - start
    return FitResult(best_metric=best_metric, train_time=train_time)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, task_type: str, device: str) -> Dict[str, float]:
    model.eval()
    ys: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    for xb, yb in loader:
        out = model(xb.to(device)).cpu().numpy()
        preds.append(out)
        ys.append(yb.numpy())
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    if task_type == "classification":
        classes = np.argmax(y_pred, axis=1)
        accuracy = float((classes == y_true).mean())
        confidence = float(np.max(torch.softmax(torch.from_numpy(y_pred), dim=1).numpy(), axis=1).mean())
        return {"primary": accuracy, "accuracy": accuracy, "confidence": confidence}
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return {"primary": mse, "mse": mse, "mae": mae}


def aggregate_seed_metrics(records: List[Dict[str, float]], key: str) -> Dict[str, float]:
    values = np.asarray([r[key] for r in records], dtype=np.float32)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }

