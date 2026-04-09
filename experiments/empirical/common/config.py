"""
Config loading and validation for paper-claims experiments.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _to_plain(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {k: _to_plain(v) for k, v in vars(value).items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


def load_config(path: str | Path) -> SimpleNamespace:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Paper-claims config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return _to_namespace(data)


def to_plain_dict(cfg: SimpleNamespace) -> dict:
    return _to_plain(cfg)


def validate_config(cfg: SimpleNamespace) -> None:
    """Validate config. Only universally-required sections are enforced;
    experiment-specific sections are validated only if present."""
    # Universal minimum: every config must have these
    required_sections = ["memory", "acceptance_gates", "plotting", "output_dir"]
    for section in required_sections:
        if not hasattr(cfg, section):
            raise ValueError(f"Missing required config section: {section}")
    if cfg.memory.K <= 0:
        raise ValueError("memory.K must be positive")

    # Conditional checks: only validate if the section exists
    if hasattr(cfg, "training"):
        if cfg.training.batch_size <= 0:
            raise ValueError("training.batch_size must be positive")
        if cfg.training.smoke_epochs <= 0 or cfg.training.full_epochs <= 0:
            raise ValueError("training epochs must be positive")
