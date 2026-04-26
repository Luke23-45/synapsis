"""
SYNAPSE Adapter — Pre-computed Features → Torch Projection
==========================================================

Projects pre-computed SYNAPSE memory features (anchor clouds and
topological summaries) into the transformer's embedding space.

CRITICAL DESIGN DECISION (Professor Feedback #2):
    M() is NOT called inside the model's forward pass. The CPU-bound
    numpy computation would starve the GPU during training. Instead,
    SYNAPSE features are pre-computed offline and cached (see synapse_cache.py).
    This adapter receives pre-computed tensors and only applies learnable
    projection layers — the sole trainable parameters added by the memory
    system (~0.8% of total parameters).

Ablation control (Professor Feedback #4):
    Boolean flags `use_anchors` and `use_topo` control which SYNAPSE
    components are included in the token sequence. This enables the
    B-Anchors and B-Topo ablation conditions without any architectural
    changes — only token masking at concatenation time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple

from src.core.config import ExperimentConfig, Condition


class SynapseAdapter(nn.Module):
    """Projects pre-computed SYNAPSE features into transformer embedding space.

    M() is called offline (see synapse_cache.py). This module only applies
    learnable projections — the sole trainable parameters added by the
    memory system.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration containing all dimensional parameters.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()

        self.use_anchors = config.condition.use_anchors
        self.use_topo = config.condition.use_topo
        self.d_model = config.transformer.d_model
        self.K = config.synapse.K
        self.Q = config.synapse.Q

        anchor_feature_dim = config.anchor_feature_dim  # proprio_dim + 3
        topo_feature_dim = config.topo_feature_dim      # 4 * (Q + 1)

        if not self.use_anchors and not self.use_topo:
            raise ValueError(
                "At least one of use_anchors/use_topo must be True. "
                "Both False produces no tokens — the adapter is vacuous."
            )

        # Learnable projection layers — the ONLY additional parameters
        if self.use_anchors:
            self.anchor_proj = nn.Linear(anchor_feature_dim, self.d_model)
            # Initialize with small weights to prevent initial dominance
            nn.init.xavier_uniform_(self.anchor_proj.weight)
            nn.init.zeros_(self.anchor_proj.bias)

        if self.use_topo:
            self.topo_proj = nn.Linear(topo_feature_dim, self.d_model)
            nn.init.xavier_uniform_(self.topo_proj.weight)
            nn.init.zeros_(self.topo_proj.bias)

    @property
    def num_output_tokens(self) -> int:
        """Number of tokens produced by this adapter.

        B-Full:     K + 1 (K anchors + 1 topology)
        B-Anchors:  K     (anchors only)
        B-Topo:     1     (topology only)
        """
        count = 0
        if self.use_anchors:
            count += self.K
        if self.use_topo:
            count += 1
        return count

    @property
    def num_trainable_params(self) -> int:
        """Count of trainable parameters added by the memory system."""
        total = 0
        if self.use_anchors:
            total += sum(p.numel() for p in self.anchor_proj.parameters())
        if self.use_topo:
            total += sum(p.numel() for p in self.topo_proj.parameters())
        return total

    def forward(
        self,
        anchor_clouds: torch.Tensor,
        topo_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Project pre-computed SYNAPSE features into embedding space.

        Parameters
        ----------
        anchor_clouds : torch.Tensor, shape (B, K, D_anchor)
            Pre-computed anchor point cloud, padded to K rows.
            Detached from any computation graph — computed offline.
        topo_features : torch.Tensor, shape (B, D_topo)
            Pre-computed topological summary vector.
            Detached from any computation graph — computed offline.

        Returns
        -------
        tokens : torch.Tensor, shape (B, N, d_model)
            Projected SYNAPSE tokens ready for transformer input.
            N = num_output_tokens (K+1, K, or 1 depending on ablation).
        n_real : int
            Number of real (non-padding) tokens. Used for attention masking.
        """
        tokens = []

        if self.use_anchors:
            # anchor_clouds: (B, K, D_anchor) → (B, K, d_model)
            anchor_tokens = self.anchor_proj(anchor_clouds)
            tokens.append(anchor_tokens)

        if self.use_topo:
            # topo_features: (B, D_topo) → (B, 1, d_model)
            topo_token = self.topo_proj(topo_features).unsqueeze(1)
            tokens.append(topo_token)

        # Concatenate along token dimension
        # B-Full: (B, K, d_model) + (B, 1, d_model) → (B, K+1, d_model)
        # B-Anchors: (B, K, d_model)
        # B-Topo: (B, 1, d_model)
        combined = torch.cat(tokens, dim=1)

        return combined, self.num_output_tokens

    def forward_with_dummy(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        """Produce zero-filled tokens for conditions that don't use SYNAPSE.

        This is used for shape testing and for conditions A1/A2 which
        have no SYNAPSE features but need to verify interface compatibility.

        Parameters
        ----------
        batch_size : int
        device : torch.device

        Returns
        -------
        tokens : torch.Tensor, shape (B, 0, d_model)
            Empty token sequence (no SYNAPSE tokens for A1/A2).
        n_real : int
            Always 0 for non-SYNAPSE conditions.
        """
        return (
            torch.zeros(batch_size, 0, self.d_model, device=device),
            0,
        )
