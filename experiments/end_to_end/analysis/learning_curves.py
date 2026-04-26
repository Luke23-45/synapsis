"""
Learning-curve analysis for end-to-end training.

This module validates the recorded history, generates plots that survive
partial-validation schedules, and writes a compact JSON summary that can be
referenced when comparing runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


def load_history(path: Path) -> List[Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_history(history: List[Dict[str, float]]) -> None:
    if not history:
        raise ValueError("Training history is empty.")
    required = {"epoch", "train_loss", "lr", "avg_grad_norm"}
    missing = [key for key in required if key not in history[0]]
    if missing:
        raise ValueError(f"Training history is missing required keys: {missing}")


def _series(history: List[Dict[str, float]], key: str) -> np.ndarray:
    values = [record.get(key, np.nan) for record in history]
    return np.asarray(values, dtype=np.float32)


def save_learning_summary(
    history: List[Dict[str, float]],
    save_dir: Path,
) -> Dict[str, float]:
    _require_history(history)

    epochs = _series(history, "epoch")
    train_loss = _series(history, "train_loss")
    val_loss = _series(history, "val_loss")
    val_mse = _series(history, "val_action_mse")
    grad_norm = _series(history, "avg_grad_norm")

    finite_val_loss = np.isfinite(val_loss)
    best_val_epoch = int(epochs[finite_val_loss][np.argmin(val_loss[finite_val_loss])]) if finite_val_loss.any() else -1
    best_val_loss = float(np.nanmin(val_loss)) if finite_val_loss.any() else float("nan")
    best_val_mse = float(np.nanmin(val_mse)) if np.isfinite(val_mse).any() else float("nan")

    summary = {
        "num_epochs": float(len(history)),
        "last_epoch": float(epochs[-1]),
        "final_train_loss": float(train_loss[-1]),
        "best_train_loss": float(np.nanmin(train_loss)),
        "best_val_epoch": float(best_val_epoch),
        "best_val_loss": best_val_loss,
        "best_val_action_mse": best_val_mse,
        "max_grad_norm": float(np.nanmax(grad_norm)),
        "mean_grad_norm": float(np.nanmean(grad_norm)),
    }

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "learning_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def plot_learning_curves(
    history: List[Dict[str, float]],
    save_dir: Path,
    title_prefix: str = "Z2 E2E",
) -> None:
    _require_history(history)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = _series(history, "epoch")
    train_loss = _series(history, "train_loss")
    train_action = _series(history, "train_action_loss")
    val_loss = _series(history, "val_loss")
    val_mse = _series(history, "val_action_mse")
    lrs = _series(history, "lr")
    grad_norms = _series(history, "avg_grad_norm")

    val_mask = np.isfinite(val_loss)
    val_epochs = epochs[val_mask]
    val_loss_clean = val_loss[val_mask]
    val_mse_clean = val_mse[np.isfinite(val_mse)]
    val_mse_epochs = epochs[np.isfinite(val_mse)]

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label="Train Loss", color="#1E88E5", linewidth=1.8)
    if val_loss_clean.size:
        ax.plot(val_epochs, val_loss_clean, label="Validation Loss", color="#E53935", linewidth=2.0, marker="o", markersize=3.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title_prefix} - Training and Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_dir / "loss_curves.png", dpi=220, bbox_inches="tight")
    fig.savefig(save_dir / "loss_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    if val_mse_clean.size:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, train_action, label="Train Action Loss", color="#3949AB", linewidth=1.5, alpha=0.8)
        ax.plot(val_mse_epochs, val_mse_clean, label="Validation Action MSE", color="#FB8C00", linewidth=2.0, marker="o", markersize=3.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title(f"{title_prefix} - Action Prediction Quality", fontsize=14, fontweight="bold")
        ax.legend(framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(save_dir / "action_mse.png", dpi=220, bbox_inches="tight")
        fig.savefig(save_dir / "action_mse.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, lrs, color="#43A047", linewidth=1.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.set_title(f"{title_prefix} - Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_dir / "lr_schedule.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, grad_norms, color="#8E24AA", linewidth=1.1, alpha=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    ax.set_title(f"{title_prefix} - Gradient Norms", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_dir / "grad_norms.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    save_learning_summary(history, save_dir)
    log.info("Saved learning-curve analysis to %s", save_dir)
