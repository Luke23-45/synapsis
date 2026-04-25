from __future__ import annotations

from pathlib import Path

import torch

from .model import SynapseArchitectureConfig, SynapseEndToEndModel


def build_model(config: SynapseArchitectureConfig) -> SynapseEndToEndModel:
    return SynapseEndToEndModel(config)


def load_model(path: str | Path, config: SynapseArchitectureConfig, map_location: str | torch.device = "cpu") -> SynapseEndToEndModel:
    model = build_model(config)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model
