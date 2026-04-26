"""
Condition A2: Uniform Subsampling (OFF, full horizon)
=====================================================

W timesteps are uniformly sampled across the full history 1:T.
This gives A2 the same temporal horizon as Condition B, but uses
naive uniform sampling instead of SYNAPSE's mathematical event-driven
anchor selection.

CRITICAL COMPARISON (Professor Feedback #1):
    B vs A2 isolates *representation quality* from *horizon access*.
    Both see the full history; only the summarization strategy differs.
    If B > A2, the improvement is attributable to the quality of
    SYNAPSE's mathematical summarization, not mere access to more data.

Architecture:
    Current Proprio (22,) -> Linear -> d_model
                                     |
    Full History (T, 22) -> Uniform Sample W steps -> Linear -> (W, d_model) -> Concat -> Transformer -> Action Head

Token count: W + 1 = Seq_Len (no padding needed).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.config import Condition, ExperimentConfig
from .planner import RoboticsPlannerBase


class PlannerUniform(RoboticsPlannerBase):
    """Condition A2: W steps uniformly sampled across full history."""

    def __init__(self, config: ExperimentConfig) -> None:
        if config.condition != Condition.A2_UNIFORM:
            raise ValueError(
                f"PlannerUniform requires condition=A2_UNIFORM, got {config.condition}"
            )
        super().__init__(config)

        self.history_window = config.data.history_window
        self.history_encoder = nn.Linear(config.data.proprio_dim, self.d_model)
        nn.init.xavier_uniform_(self.history_encoder.weight)
        nn.init.zeros_(self.history_encoder.bias)

    @property
    def num_history_params(self) -> int:
        return sum(
            p.numel() for p in self.history_encoder.parameters() if p.requires_grad
        )

    def _compute_uniform_indices(
        self,
        T: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute W uniformly-spaced indices across [0, T-1]."""
        if T <= W:
            return torch.arange(T, device=device, dtype=torch.long)

        indices = torch.linspace(0, T - 1, W, device=device).round().long()
        indices = torch.unique(indices)

        if indices.shape[0] < W:
            all_indices = torch.arange(T, device=device, dtype=torch.long)
            remaining = all_indices[~torch.isin(all_indices, indices)]
            needed = W - indices.shape[0]
            extra = remaining[-needed:] if remaining.shape[0] >= needed else remaining
            indices = torch.cat([indices, extra])
            indices, _ = torch.sort(indices)

        return indices[:W]

    def _compute_batched_uniform_indices(
        self,
        history_lengths: torch.Tensor,
        padded_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute uniform indices over each sample's non-padded history."""
        batch_indices = []
        for hist_len in history_lengths.tolist():
            real_len = max(1, int(hist_len))
            local = self._compute_uniform_indices(real_len, self.history_window, device)
            left_pad = padded_length - real_len
            shifted = local + left_pad
            if shifted.shape[0] < self.history_window:
                pad = torch.full(
                    (self.history_window - shifted.shape[0],),
                    left_pad,
                    dtype=torch.long,
                    device=device,
                )
                shifted = torch.cat([pad, shifted], dim=0)
            batch_indices.append(shifted)
        return torch.stack(batch_indices, dim=0)

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass for Condition A2."""
        curr = self.proprio_encoder(batch["proprio"]).unsqueeze(1)

        hist_full = batch["proprio_history"]
        padded_length = hist_full.shape[1]
        history_lengths = batch.get("history_length")
        if history_lengths is None:
            history_lengths = torch.full(
                (hist_full.shape[0],),
                padded_length,
                dtype=torch.long,
                device=hist_full.device,
            )
        else:
            history_lengths = history_lengths.to(hist_full.device)

        expanded_indices = self._compute_batched_uniform_indices(
            history_lengths,
            padded_length,
            hist_full.device,
        ).unsqueeze(-1).expand(-1, -1, hist_full.shape[-1])
        hist_sampled = torch.gather(hist_full, 1, expanded_indices)

        hist_tokens = self.history_encoder(hist_sampled)
        tokens = torch.cat([curr, hist_tokens], dim=1)

        return self._apply_transformer(tokens, n_real=self.history_window + 1)
