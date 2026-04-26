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
    Current Proprio (22,) → Linear → d_model
                                         ↓
    Full History (T, 22) → Uniform Sample W steps → Linear → (W, d_model) → Concat → Transformer → Action Head

Token count: W + 1 = Seq_Len (no padding needed).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.config import ExperimentConfig, Condition
from .planner import RoboticsPlannerBase


class PlannerUniform(RoboticsPlannerBase):
    """Condition A2: W steps uniformly sampled across full history.

    Parameters
    ----------
    config : ExperimentConfig
        Must have condition = Condition.A2_UNIFORM.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        if config.condition != Condition.A2_UNIFORM:
            raise ValueError(
                f"PlannerUniform requires condition=A2_UNIFORM, got {config.condition}"
            )
        super().__init__(config)

        self.history_window = config.data.history_window

        # History encoder: per-timestep linear projection (same structure as A1)
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
        """Compute W uniformly-spaced indices across [0, T-1].

        Uses linspace to produce indices at positions:
            index[i] = round(i * (T - 1) / (W - 1))  for i = 0..W-1

        This ensures coverage of the full temporal horizon from
        the first observation to the most recent.

        Parameters
        ----------
        T : int
            Full history length.
        W : int
            Number of samples (history window).
        device : torch.device

        Returns
        -------
        indices : torch.Tensor, shape (W,), dtype long
        """
        if T <= W:
            # History shorter than window — use all available + pad
            return torch.arange(T, device=device, dtype=torch.long)

        # Uniform sampling: W evenly-spaced indices from 0 to T-1
        indices = torch.linspace(0, T - 1, W, device=device)
        indices = indices.round().long()

        # Ensure no duplicate indices (can happen when T is close to W)
        indices = torch.unique(indices)

        # If we lost indices due to dedup, pad from the end
        if indices.shape[0] < W:
            max_idx = T - 1
            all_indices = torch.arange(T, device=device, dtype=torch.long)
            remaining = all_indices[~torch.isin(all_indices, indices)]
            needed = W - indices.shape[0]
            if remaining.shape[0] >= needed:
                extra = remaining[-needed:]
            else:
                extra = remaining
            indices = torch.cat([indices, extra])
            indices, _ = torch.sort(indices)

        return indices[:W]

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass for Condition A2.

        Parameters
        ----------
        batch : dict with keys:
            proprio : torch.Tensor, shape (B, D_proprio)
                Current proprioception.
            proprio_history : torch.Tensor, shape (B, T, D_proprio)
                Full proprioceptive history. Uniformly subsampled to W steps.

        Returns
        -------
        predicted_actions : torch.Tensor, shape (B, action_chunk_size, action_dim)
        """
        # Current proprioception token
        curr = self.proprio_encoder(batch["proprio"]).unsqueeze(1)  # (B, 1, D)

        # Full history
        hist_full = batch["proprio_history"]  # (B, T, D_proprio)
        T = hist_full.shape[1]

        # Uniform subsampling across full horizon
        indices = self._compute_uniform_indices(T, self.history_window, hist_full.device)

        # Gather sampled timesteps
        # indices: (W,) → expand to (B, W) for batch gather
        expanded_indices = indices.unsqueeze(0).expand(hist_full.shape[0], -1)
        expanded_indices = expanded_indices.unsqueeze(-1).expand(
            -1, -1, hist_full.shape[-1]
        )  # (B, W, D_proprio)
        hist_sampled = torch.gather(hist_full, 1, expanded_indices)  # (B, W, D_proprio)

        # If T < W, pad the sampled history
        if hist_sampled.shape[1] < self.history_window:
            pad_len = self.history_window - hist_sampled.shape[1]
            pad = hist_full[:, :1, :].expand(-1, pad_len, -1)
            hist_sampled = torch.cat([pad, hist_sampled], dim=1)

        # Encode history
        hist_tokens = self.history_encoder(hist_sampled)  # (B, W, d_model)

        # Concatenate: [current, sampled_1, ..., sampled_W]
        tokens = torch.cat([curr, hist_tokens], dim=1)  # (B, W+1, d_model)

        # W+1 = Seq_Len, so no padding needed
        n_real = self.history_window + 1
        return self._apply_transformer(tokens, n_real=n_real)
