"""
Auxiliary losses for Z2 end-to-end training.

1. Sparsity regularization on y* — encourages selective event retention
2. Topology regularization — prevents topology branch collapse

Both are weighted by scheduled coefficients (ramped from 0 over early epochs).
"""

from __future__ import annotations

import torch


def sparsity_loss(y_star: torch.Tensor) -> torch.Tensor:
    """Encourage the relaxed selector to be selective.

    Without this regularization, the selector may spread activation
    uniformly across all timesteps, defeating the purpose of event-sparse
    memory compression.

    Parameters
    ----------
    y_star : torch.Tensor
        Shape (B, T). Relaxed selector activations in [0, 1].

    Returns
    -------
    torch.Tensor
        Scalar loss: mean L1 norm of y* per sequence, normalized by T.
    """
    # L1 norm per sequence, normalized by sequence length
    T = y_star.shape[1]
    per_seq_l1 = y_star.abs().sum(dim=1) / max(T, 1)  # (B,)
    return per_seq_l1.mean()


def topology_reg_loss(topology_token: torch.Tensor) -> torch.Tensor:
    """Prevent topology branch from collapsing to a constant.

    If the topology branch outputs the same vector for every sequence
    in the batch, the variance across the batch dimension is zero and
    this loss diverges → strong corrective gradient signal.

    Parameters
    ----------
    topology_token : torch.Tensor
        Shape (B, d_model). Topology features for each sequence in batch.

    Returns
    -------
    torch.Tensor
        Scalar loss: -log(mean variance across features + eps).
    """
    if topology_token.shape[0] < 2:
        # Cannot compute variance with a single sample
        return torch.tensor(0.0, device=topology_token.device, dtype=topology_token.dtype)

    # Variance across batch dimension for each feature, then mean
    feature_var = topology_token.var(dim=0).mean()  # scalar
    return -torch.log(feature_var + 1e-6)
