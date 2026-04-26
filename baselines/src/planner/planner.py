"""
Shared Transformer Backbone — Robotics Planner Base
====================================================

Implements the shared transformer backbone with token-count equalization
(Professor Feedback #5). All conditions feed the transformer a fixed
sequence length Seq_Len, with attention masking for padding positions.

Architecture:
    Input → Proprio Encoder → [Memory Conditioning] → Transformer → Action Head → Predicted Actions

The base class handles:
    - Proprio encoding (Linear projection)
    - Positional encoding (Learnable, shared)
    - Token padding to fixed Seq_Len with learnable <PAD>
    - Attention mask generation for padding positions
    - Transformer forward pass
    - Action head projection

Subclasses (planner_recent, planner_uniform, planner_synapse) implement
condition-specific token construction before delegating to _apply_transformer.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from src.core.config import ExperimentConfig


class RoboticsPlannerBase(nn.Module):
    """Shared backbone for all experimental conditions.

    Parameters
    ----------
    config : ExperimentConfig
        Complete experiment configuration.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()

        self.config = config
        self.d_model = config.transformer.d_model
        self.seq_len = config.seq_len  # W + 1 (§3.6)
        self.proprio_dim = config.data.proprio_dim
        self.action_chunk_size = config.data.action_chunk_size
        self.action_dim = config.data.action_dim

        # Proprio encoder: maps raw proprioception to d_model
        self.proprio_encoder = nn.Linear(self.proprio_dim, self.d_model)

        # Learnable positional encoding — shared across conditions
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.seq_len, self.d_model) * 0.02
        )

        # Learnable padding token for Seq_Len equalization (§3.6)
        self.pad_token = nn.Parameter(
            torch.randn(1, 1, self.d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.transformer.num_heads,
            dim_feedforward=self.d_model * config.transformer.ffn_ratio,
            dropout=config.transformer.dropout,
            activation=config.transformer.activation,
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.transformer.num_layers,
            enable_nested_tensor=False,
        )

        # Action head: maps transformer output to action chunks
        self.action_head = nn.Linear(
            self.d_model,
            self.action_chunk_size * self.action_dim,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with careful scaling."""
        nn.init.xavier_uniform_(self.proprio_encoder.weight)
        nn.init.zeros_(self.proprio_encoder.bias)
        nn.init.xavier_uniform_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)

    @property
    def num_trainable_params(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _build_key_padding_mask(
        self,
        n_real: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build boolean key padding mask for the transformer.

        True = padding position (ignored by attention).
        False = real position (attended to normally).

        Parameters
        ----------
        n_real : int
            Number of real (non-padding) tokens in the sequence.
        batch_size : int
        device : torch.device

        Returns
        -------
        mask : torch.Tensor, shape (B, seq_len)
            Boolean key padding mask.
        """
        mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool, device=device)
        if n_real < self.seq_len:
            mask[:, n_real:] = True
        return mask

    def _apply_transformer(
        self,
        tokens: torch.Tensor,
        n_real: int,
    ) -> torch.Tensor:
        """Pad tokens to Seq_Len, apply positional encoding, run transformer,
        and produce action predictions.

        Parameters
        ----------
        tokens : torch.Tensor, shape (B, L, d_model)
            Real token sequence. L ≤ seq_len.
        n_real : int
            Number of real tokens (used for attention masking).

        Returns
        -------
        predicted_actions : torch.Tensor, shape (B, action_chunk_size, action_dim)
        """
        B, L, D = tokens.shape
        device = tokens.device

        # Pad to fixed Seq_Len if needed
        if L < self.seq_len:
            pad = self.pad_token.expand(B, self.seq_len - L, -1)
            tokens = torch.cat([tokens, pad], dim=1)
        elif L > self.seq_len:
            # This should never happen — indicates a bug
            raise ValueError(
                f"Token sequence length {L} exceeds Seq_Len {self.seq_len}. "
                f"This is a programming error in the condition-specific "
                f"token construction."
            )

        # Add positional encoding
        tokens = tokens + self.pos_embed[:, :self.seq_len, :]

        # Build key padding mask for padding positions
        key_padding_mask = self._build_key_padding_mask(n_real, B, device)

        # Transformer forward pass
        # No causal mask — this is not autoregressive generation.
        # All real positions can attend to all other real positions.
        # Padding positions are masked via src_key_padding_mask.
        out = self.transformer(tokens, src_key_padding_mask=key_padding_mask)

        # Predict from the first token (current proprioception)
        first_token_out = out[:, 0, :]  # (B, d_model)
        action_flat = self.action_head(first_token_out)  # (B, K_act * D_act)

        # Reshape to (B, action_chunk_size, action_dim)
        predicted_actions = action_flat.view(
            B, self.action_chunk_size, self.action_dim
        )

        return predicted_actions

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass — must be implemented by subclasses.

        Parameters
        ----------
        batch : dict
            Batch dictionary with condition-specific keys.

        Returns
        -------
        predicted_actions : torch.Tensor, shape (B, action_chunk_size, action_dim)
        """
        raise NotImplementedError(
            "Subclasses must implement forward() to construct condition-specific "
            "tokens before calling _apply_transformer()."
        )

    def count_parameters_by_component(self) -> dict:
        """Count trainable parameters per component for verification."""
        counts = {}
        counts["proprio_encoder"] = sum(
            p.numel() for p in self.proprio_encoder.parameters() if p.requires_grad
        )
        counts["pos_embed"] = self.pos_embed.numel()
        counts["pad_token"] = self.pad_token.numel()
        counts["transformer"] = sum(
            p.numel() for p in self.transformer.parameters() if p.requires_grad
        )
        counts["action_head"] = sum(
            p.numel() for p in self.action_head.parameters() if p.requires_grad
        )
        counts["total_base"] = sum(counts.values())
        return counts
