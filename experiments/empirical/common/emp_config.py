"""
experiments/empirical/common/emp_config.py

Config loader for Phase 2 EMP-* experiments.
Merges base.yaml + per-experiment YAML into a single SimpleNamespace.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import yaml


_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "empirical"
_BASE_YAML = _CONFIGS_DIR / "base.yaml"

# Map experiment IDs to their config files
_EMP_YAMLS: Dict[str, str] = {
    "EMP-01": "emp01_event_encoder.yaml",
    "EMP-02": "emp02_selector_gradient.yaml",
    "EMP-03": "emp03_lift_training.yaml",
    "EMP-04": "emp04_topology_value.yaml",
    "EMP-05": "emp05_train_deploy.yaml",
    "EMP-06": "emp06_memory_sufficiency.yaml",
    "EMP-07": "emp07_scalability.yaml",
    "EMP-08": "emp08_sensitivity.yaml",
    "EMP-09": "emp09_metric_structure.yaml",
    # Gap-closing experiments (Phase 2 audit, April 2026)
    "EMP-10": "emp10_causality.yaml",
    "EMP-11": "emp11_topological_stability.yaml",
    "EMP-12": "emp12_exact_recovery.yaml",
    "EMP-13": "emp13_hard_projection.yaml",
    # Legacy EZ2 experiments (foundation/robotics) — modernized
    "EZ2-01": "ez01_anchor_alignment.yaml",
    "EZ2-03": "ez03_event_recovery.yaml",
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _to_namespace(value: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _to_dict(value: Any) -> Any:
    """Recursively convert SimpleNamespace to dict."""
    if isinstance(value, SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(value).items()}
    if isinstance(value, list):
        return [_to_dict(v) for v in value]
    return value


def load_emp_config(
    experiment_id: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> SimpleNamespace:
    """
    Load config for an EMP experiment by merging base.yaml + experiment YAML.

    Parameters
    ----------
    experiment_id : str
        e.g. "EMP-01", "EMP-02", ...
    overrides : dict, optional
        Runtime overrides applied last (e.g. from CLI).

    Returns
    -------
    SimpleNamespace
        Merged config with all sections accessible as attributes.
    """
    # Load base
    if _BASE_YAML.exists():
        base_data = yaml.safe_load(_BASE_YAML.read_text(encoding="utf-8")) or {}
    else:
        base_data = {}

    # Load experiment-specific
    exp_filename = _EMP_YAMLS.get(experiment_id)
    exp_data: dict = {}
    if exp_filename:
        exp_path = _CONFIGS_DIR / exp_filename
        if exp_path.exists():
            exp_data = yaml.safe_load(exp_path.read_text(encoding="utf-8")) or {}

    # Merge: base <- experiment <- runtime overrides
    merged = _deep_merge(base_data, exp_data)
    if overrides:
        merged = _deep_merge(merged, overrides)

    cfg = _to_namespace(merged)

    # Ensure critical attributes exist with safe defaults
    if not hasattr(cfg, "hardware"):
        cfg.hardware = SimpleNamespace(
            accelerator="auto", devices=1, precision=32,
            num_workers=0, pin_memory=True,
        )
    if not hasattr(cfg, "training"):
        cfg.training = SimpleNamespace(
            batch_size=32, lr=1e-3, weight_decay=1e-4,
            smoke_epochs=5, full_epochs=50, patience=10,
            num_seeds=3, seeds=[42, 123, 456],
        )
    if not hasattr(cfg, "memory"):
        cfg.memory = SimpleNamespace(K=10, r=2, lam=1.0, k=8, Q=1, solver="scipy")
    if not hasattr(cfg, "trajectory"):
        cfg.trajectory = SimpleNamespace(d=5, T=100)
    if not hasattr(cfg, "data"):
        cfg.data = SimpleNamespace(n_train=200, n_val=50, n_test=100, noise_std=0.2)
    if not hasattr(cfg, "output_dir"):
        cfg.output_dir = "experiments/outputs/empirical"

    # Ensure seeds is always a list
    if hasattr(cfg.training, "seeds") and not isinstance(cfg.training.seeds, list):
        cfg.training.seeds = [cfg.training.seeds]

    return cfg


def get_device_config(cfg: SimpleNamespace) -> Dict[str, Any]:
    """
    Extract Lightning Trainer kwargs from cfg.hardware.

    Returns
    -------
    dict suitable for pl.Trainer(**kwargs)
    """
    hw = cfg.hardware
    return {
        "accelerator": getattr(hw, "accelerator", "auto"),
        "devices": getattr(hw, "devices", 1),
        "precision": getattr(hw, "precision", 32),
    }


def get_dataloader_kwargs(cfg: SimpleNamespace) -> Dict[str, Any]:
    """Extract DataLoader kwargs from cfg.hardware."""
    hw = cfg.hardware
    kwargs: Dict[str, Any] = {
        "num_workers": getattr(hw, "num_workers", 0),
    }
    if getattr(hw, "pin_memory", False):
        kwargs["pin_memory"] = True
    return kwargs
