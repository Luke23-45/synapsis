"""
Condition A1: Recent Window (OFF, horizon-limited)
==================================================

Uses the last W timesteps of proprioception as history tokens.
No SYNAPSE computation. No topological features. No anchor selection.

This condition is horizon-limited: it cannot see beyond W steps into
the past. It represents the standard sliding-window approach used by
most transformer-based robotics planners.

Architecture:
    Current Proprio (22,) → Linear → d_model
                                         ↓
    History Window (W, 22) → Linear → (W, d_model) → Concat → Transformer → Action Head

Token count: W + 1 = Seq_Len (no padding needed).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.config import ExperimentConfig, Condition
from .planner import RoboticsPlannerBase


class PlannerRecent(RoboticsPlannerBase):
    """Condition A1: Last W steps of history.

    Parameters
    ----------
    config : ExperimentConfig
        Must have condition = Condition.A1_RECENT.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        if config.condition != Condition.A1_RECENT:
            raise ValueError(
                f"PlannerRecent requires condition=A1_RECENT, got {config.condition}"
            )
        super().__init__(config)

        self.history_window = config.data.history_window

        # History encoder: per-timestep linear projection
        self.history_encoder = nn.Linear(config.data.proprio_dim, self.d_model)
        nn.init.xavier_uniform_(self.history_encoder.weight)
        nn.init.zeros_(self.history_encoder.bias)

    @property
    def num_history_params(self) -> int:
        return sum(
            p.numel() for p in self.history_encoder.parameters() if p.requires_grad
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass for Condition A1.

        Parameters
        ----------
        batch : dict with keys:
            proprio : torch.Tensor, shape (B, D_proprio)
                Current proprioception.
            proprio_history : torch.Tensor, shape (B, T, D_proprio)
                Full proprioceptive history. Only the last W steps are used.

        Returns
        -------
        predicted_actions : torch.Tensor, shape (B, action_chunk_size, action_dim)
        """
        # Current proprioception token
        curr = self.proprio_encoder(batch["proprio"]).unsqueeze(1)  # (B, 1, D)

        # History window: last W steps
        hist_full = batch["proprio_history"]  # (B, T, D_proprio)
        T = hist_full.shape[1]

        if T >= self.history_window:
            hist_window = hist_full[:, -self.history_window:, :]  # (B, W, D)
        else:
            # Pad with first observation if episode shorter than W
            pad_len = self.history_window - T
            pad = hist_full[:, :1, :].expand(-1, pad_len, -1)  # (B, pad_len, D)
            hist_window = torch.cat([pad, hist_full], dim=1)  # (B, W, D)

        # Encode history
        hist_tokens = self.history_encoder(hist_window)  # (B, W, d_model)

        # Concatenate: [current, history_1, ..., history_W]
        tokens = torch.cat([curr, hist_tokens], dim=1)  # (B, W+1, d_model)

        # W+1 = Seq_Len, so no padding needed
        n_real = self.history_window + 1
        return self._apply_transformer(tokens, n_real=n_real)
