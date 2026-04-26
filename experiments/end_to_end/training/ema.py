"""
Exponential Moving Average (EMA) of model weights.

Maintains a shadow copy of model parameters updated each training step:
  θ_ema = decay · θ_ema + (1 - decay) · θ

The EMA model is used for evaluation to smooth out training noise.
Standard in SOTA robotics policies (ACT, Diffusion Policy).
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMAModel:
    """Exponential Moving Average wrapper for any nn.Module.

    Parameters
    ----------
    model : nn.Module
        The model to track. Parameters are copied at initialization.
    decay : float
        EMA decay factor. Higher = smoother, slower to adapt.
        Typical values: 0.999 (stable), 0.9999 (very smooth).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        # Deep copy of model parameters (detached, on same device)
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()
        # Also track buffers (e.g., NormalizedLift.mu, sigma)
        self.shadow_buffers: dict[str, torch.Tensor] = {}
        for name, buf in model.named_buffers():
            self.shadow_buffers[name] = buf.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters from the current model state.

        Should be called once per training step, after optimizer.step().
        """
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_to(self, model: nn.Module) -> None:
        """Copy EMA parameters into the model (for evaluation).

        After calling this, the model's parameters will be the EMA-smoothed
        version. To restore original parameters, use swap_with().
        """
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
        for name, buf in model.named_buffers():
            if name in self.shadow_buffers:
                buf.data.copy_(self.shadow_buffers[name])

    def state_dict(self) -> dict:
        """Serialize EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
            "shadow_buffers": {k: v.cpu() for k, v in self.shadow_buffers.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore EMA state from checkpoint."""
        self.decay = state["decay"]
        for k, v in state["shadow"].items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device))
        if "shadow_buffers" in state:
            for k, v in state["shadow_buffers"].items():
                if k in self.shadow_buffers:
                    self.shadow_buffers[k].copy_(v.to(self.shadow_buffers[k].device))
