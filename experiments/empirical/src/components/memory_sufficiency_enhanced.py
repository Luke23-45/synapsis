"""
EMP-06: Enhanced Memory Sufficiency Probe

Extends the existing EZ2-04 memory sufficiency probe with:
  (a) Neural network baselines (GRU, small Transformer) trained via Lightning
  (b) Multi-seed statistical reporting
  (c) Ridge probes on Z2 features

Spec: docs/implementation/phase2_empirical_validation/06_memory_sufficiency.md
Z2 Reference: §13 of 02_rigorous_architecture.md, Prop 12.4
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
try:
    import pytorch_lightning as pl
except ImportError:
    pl = None
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from synapse_core.memory_operator import compute_memory
from experiments.empirical.common.baselines import (
    summarize_diagrams, uniform_feature, GRUHead, TransformerHead,
)
from experiments.empirical.common.tasks import generate_memory_task_dataset, SequenceSample
from experiments.empirical.common.seed_runner import run_multi_seed
from experiments.empirical.common.emp_config import load_emp_config, get_device_config, get_dataloader_kwargs
from experiments.empirical.common.data_saver import save_experiment_npz
from experiments.empirical.common.lightning_utils import BaseExperimentModule
from experiments.empirical.common.math_utils import (
    make_orthogonal_W,
    pad_rows,
    cloud_geometry_summary,
    ridge_probe_accuracy,
)

log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

# Helpers imported from experiments.empirical.common.math_utils:
#   make_orthogonal_W, pad_rows, cloud_geometry_summary, ridge_probe_accuracy


# ── Feature Extraction ─────────────────────────────────────────────────────

def _extract_feature(sample, baseline, K, r, lam, k, Q, W_Theta, state_cache):
    """Extract feature vector for one sample under one baseline."""
    seq = sample.sequence.astype(np.float64)
    cache_key = id(sample)

    if baseline == "B0":
        return sample.sequence[-K:].astype(np.float32).reshape(-1)
    elif baseline == "B3":
        return uniform_feature(sample.sequence, K)
    elif baseline in {"B4", "B5", "B6", "B7"}:
        if cache_key not in state_cache:
            state_cache[cache_key] = compute_memory(
                seq, K=K, r=r, lam=lam, W_Theta=W_Theta, Q=Q, solver="scipy",
            )
        state = state_cache[cache_key]
        cloud = state.point_cloud.astype(np.float32) if state.point_cloud.size else np.zeros((0, k), dtype=np.float32)
        base = pad_rows(cloud, K).reshape(-1)
        topo = summarize_diagrams(state.persistence_diagrams)
        if baseline == "B4":
            return base
        elif baseline == "B5":
            return np.concatenate([base, cloud_geometry_summary(cloud)]).astype(np.float32)
        elif baseline == "B6":
            return np.concatenate([base, topo]).astype(np.float32)
        else:  # B7
            from synapse_core.anchor_selector import build_anchors
            from synapse_core.geometric_lift import anchor_vectors, normalize_anchors, apply_lift
            indices = [idx for idx, w in enumerate(state.y_star) if w > 0.01]
            if not indices:
                return np.zeros(K * k, dtype=np.float32)
            anchors = build_anchors(indices, seq, state.event_scores)
            V = anchor_vectors(anchors, D=seq.shape[1] + 3)
            V_norm, _, _ = normalize_anchors(V)
            lifted = apply_lift(V_norm, W_Theta)
            weighted = lifted * state.y_star[indices][:, None]
            return pad_rows(weighted.astype(np.float32), K).reshape(-1)
    raise KeyError(f"Unknown baseline: {baseline}")


# ── Ridge Probe ────────────────────────────────────────────────────────────

def _ridge_probe_accuracy_impl(X_train, y_train, X_val, y_val, X_test, y_test):
    """Ridge probe with validation — delegates to math_utils."""
    return ridge_probe_accuracy(
        X_train.astype(np.float64), y_train,
        X_test.astype(np.float64), y_test,
        X_val=X_val.astype(np.float64), y_val=y_val,
    )


# ── Neural Baseline Training via Lightning ─────────────────────────────────

def _train_neural_baseline(
    model: nn.Module,
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor,
    X_test: torch.Tensor, y_test: torch.Tensor,
    n_classes: int,
    epochs: int, lr: float, bs: int,
    device_cfg: Dict[str, Any],
    dl_kwargs: Dict[str, Any],
) -> float:
    """Train a neural baseline (GRU/Transformer) with Lightning and return test accuracy."""
    lit = BaseExperimentModule(
        model=model, task_type="classification",
        lr=lr, num_classes=n_classes,
    )
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=bs, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=bs, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=bs, shuffle=False, **dl_kwargs)

    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val/loss", patience=8, mode="min"),
        pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1),
    ]
    trainer = pl.Trainer(
        max_epochs=epochs, callbacks=callbacks,
        enable_progress_bar=False, enable_model_summary=False, logger=False,
        **device_cfg,
    )
    trainer.fit(lit, train_loader, val_loader)
    results = trainer.test(lit, test_loader, ckpt_path="best", verbose=False)
    return results[0].get("test/acc", 0.0) if results else 0.0


# ── Single-Seed Experiment ─────────────────────────────────────────────────

def run_single_seed(config: Any, seed: int) -> Dict[str, float]:
    """Run EMP-06 for a single seed."""
    if pl is not None:
        pl.seed_everything(seed, workers=True)
    cfg = config
    rng = np.random.default_rng(seed)

    K = cfg.memory.K
    r = cfg.memory.r
    lam = cfg.memory.lam
    k = cfg.memory.k
    Q = cfg.memory.Q
    d = cfg.trajectory.d
    T = cfg.trajectory.T
    n_train = cfg.data.n_train
    n_val = cfg.data.n_val
    n_test = cfg.data.n_test
    bs = cfg.training.batch_size
    neural_epochs = getattr(cfg.training, "neural_epochs", cfg.training.full_epochs)
    lr = cfg.training.lr

    families = cfg.task_families if hasattr(cfg, "task_families") else [
        "delayed_retrieval", "ordered_trigger_response",
        "phase_conditioned_decision", "revisit_dependent_rule_switching",
    ]
    ridge_baselines = cfg.ridge_baselines if hasattr(cfg, "ridge_baselines") else ["B0", "B3", "B4", "B5", "B6", "B7"]

    D = d + 3
    W_Theta = make_orthogonal_W(k, D, rng)
    device_cfg = get_device_config(cfg)
    dl_kwargs = get_dataloader_kwargs(cfg)

    results: Dict[str, float] = {}

    for family in families:
        train_data = generate_memory_task_dataset(rng, family, n_train, T, d, 0.25)
        val_data = generate_memory_task_dataset(rng, family, max(20, n_val), T, d, 0.35)
        test_data = generate_memory_task_dataset(rng, family, max(30, n_test), T, d, 0.55)

        all_labels = set(s.label for s in train_data + val_data + test_data)
        n_classes = max(len(all_labels), 2)
        y_train = np.array([s.label for s in train_data], dtype=np.int64)
        y_val = np.array([s.label for s in val_data], dtype=np.int64)
        y_test = np.array([s.label for s in test_data], dtype=np.int64)

        state_cache: Dict = {}

        # ── Ridge baselines (B0-B7) ──
        for bname in ridge_baselines:
            try:
                X_tr = np.stack([_extract_feature(s, bname, K, r, lam, k, Q, W_Theta, state_cache) for s in train_data]).astype(np.float64)
                X_va = np.stack([_extract_feature(s, bname, K, r, lam, k, Q, W_Theta, state_cache) for s in val_data]).astype(np.float64)
                X_te = np.stack([_extract_feature(s, bname, K, r, lam, k, Q, W_Theta, state_cache) for s in test_data]).astype(np.float64)
                results[f"{family}/{bname}/ridge"] = _ridge_probe_accuracy_impl(X_tr, y_train, X_va, y_val, X_te, y_test)
            except Exception as exc:
                log.warning("Feature extraction failed %s/%s: %s", family, bname, exc)
                results[f"{family}/{bname}/ridge"] = 0.0

        # ── Neural baselines via Lightning (GPU-accelerated) ──
        max_T = max(s.sequence.shape[0] for s in train_data + val_data + test_data)

        def _pad_seqs(samples):
            padded = torch.zeros(len(samples), max_T, d)
            for i, s in enumerate(samples):
                L = s.sequence.shape[0]
                padded[i, :L] = torch.from_numpy(s.sequence).float()
            return padded

        X_seq_train = _pad_seqs(train_data)
        X_seq_val = _pad_seqs(val_data)
        X_seq_test = _pad_seqs(test_data)
        y_train_t = torch.tensor(y_train).long()
        y_val_t = torch.tensor(y_val).long()
        y_test_t = torch.tensor(y_test).long()

        for neural_name, ModelClass in [("B8_GRU", GRUHead), ("B9_Trans", TransformerHead)]:
            try:
                pl.seed_everything(seed, workers=True)
                model = ModelClass(input_dim=d, hidden_dim=64, output_dim=n_classes)
                acc = _train_neural_baseline(
                    model, X_seq_train, y_train_t, X_seq_val, y_val_t,
                    X_seq_test, y_test_t, n_classes,
                    neural_epochs, lr, bs, device_cfg, dl_kwargs,
                )
                results[f"{family}/{neural_name}"] = acc
            except Exception as exc:
                log.warning("Neural baseline failed %s/%s: %s", family, neural_name, exc)
                results[f"{family}/{neural_name}"] = 0.0

    # ── Summary metrics ──
    b6_accs = [v for k, v in results.items() if "/B6/ridge" in k]
    b8_accs = [v for k, v in results.items() if "/B8_GRU" in k]
    results["B6_mean_across_families"] = float(np.mean(b6_accs)) if b6_accs else 0.0
    results["B8_GRU_mean_across_families"] = float(np.mean(b8_accs)) if b8_accs else 0.0

    margin = 0.05
    if hasattr(cfg, "acceptance_gates"):
        margin = getattr(cfg.acceptance_gates, "b6_vs_b8_margin", margin)

    competitive = sum(
        1 for fam in families
        if results.get(f"{fam}/B6/ridge", 0.0) >= results.get(f"{fam}/B8_GRU", 0.0) - margin
    )
    results["competitive_families"] = float(competitive)
    results["total_families"] = float(len(families))
    
    save_experiment_npz("EMP-06", seed, {
        "test_sequences": np.stack([s.sequence for s in test_data]),
        "test_labels": np.array([s.label for s in test_data])
    }, cfg.output_dir)

    return results


# ── Entry Point ────────────────────────────────────────────────────────────

def run_experiment(config: Any = None) -> Dict[str, Any]:
    """Run EMP-06 across all seeds."""
    if config is None:
        config = load_emp_config("EMP-06")

    report = run_multi_seed(
        experiment_fn=run_single_seed,
        config=config,
        seeds=config.training.seeds,
        experiment_id="EMP-06",
        output_dir=config.output_dir,
    )

    agg = report["aggregated"]
    competitive = agg.get("competitive_families", {}).get("mean", 0.0)
    total = agg.get("total_families", {}).get("mean", 1.0)
    min_ratio = 0.50
    if hasattr(config, "acceptance_gates"):
        min_ratio = getattr(config.acceptance_gates, "min_competitive_families", min_ratio)
    passed = competitive >= total * min_ratio

    report["experiment_name"] = "Enhanced Memory Sufficiency"
    report["pass_criterion"] = f"competitive >= {min_ratio * 100:.0f}% of families"
    report["passed"] = passed

    b6 = agg.get("B6_mean_across_families", {}).get("mean", 0.0)
    b8 = agg.get("B8_GRU_mean_across_families", {}).get("mean", 0.0)
    log.info("[EMP-06] B6=%.4f  B8_GRU=%.4f  competitive=%d/%d  passed=%s",
             b6, b8, int(competitive), int(total), passed)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if str(Path(__file__).resolve().parent.parent.parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
    result = run_experiment()
    print(f"\nEMP-06 PASSED: {result['passed']}")
