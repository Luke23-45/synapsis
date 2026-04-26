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


def topology_reg_loss(topology_token: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    """Prevent topology branch from collapsing to a constant.

    Uses a Hinge loss on the standard deviation (similar to VICReg) to
    maintain a healthy feature variance across the batch. This avoids the
    violent gradient explosions (1e6+) caused by -log(x) when var -> 0.

    Parameters
    ----------
    topology_token : torch.Tensor
        Shape (B, d_model). Topology features for each sequence in batch.
    target_std : float
        The target standard deviation to maintain across the batch dimension.

    Returns
    -------
    torch.Tensor
        Scalar loss: mean(ReLU(target_std - std_across_batch)).
    """
    if topology_token.shape[0] < 2:
        return torch.tensor(0.0, device=topology_token.device, dtype=topology_token.dtype)

    # Variance across batch dimension, then std dev
    # Add eps inside sqrt for numerical stability
    var = topology_token.var(dim=0)
    std = torch.sqrt(var + 1e-4)
    
    # Hinge loss: pushes std up to target_std, then stops
    # Gradient is bounded to exactly -1.0 (or 0), eliminating explosion
    return torch.mean(torch.relu(target_std - std))
