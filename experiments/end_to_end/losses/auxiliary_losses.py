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


def topology_reg_loss(
    topology_token: torch.Tensor,
    target_std: float = 0.25,
    covariance_weight: float = 0.05,
) -> torch.Tensor:
    """Prevent topology branch from collapsing to a constant.

    Uses a VICReg-style variance floor plus a lightweight covariance penalty.
    The previous target_std=1.0 was unrealistically large for this branch's
    initialization scale, so the loss stayed pinned near 1.0 and conveyed
    little information in logs.

    Parameters
    ----------
    topology_token : torch.Tensor
        Shape (B, d_model). Topology features for each sequence in batch.
    target_std : float
        Target standard deviation across the batch dimension.
    covariance_weight : float
        Weight on off-diagonal covariance energy to discourage collapsed,
        redundant topology channels.

    Returns
    -------
    torch.Tensor
        Scalar loss: mean(ReLU(target_std - std_across_batch)).
    """
    if topology_token.shape[0] < 2:
        return torch.tensor(0.0, device=topology_token.device, dtype=topology_token.dtype)

    centered = topology_token - topology_token.mean(dim=0, keepdim=True)
    var = centered.var(dim=0, unbiased=False)
    std = torch.sqrt(var + 1e-4)
    variance_term = torch.mean(torch.relu(target_std - std))

    feature_dim = topology_token.shape[1]
    cov = centered.T @ centered / max(topology_token.shape[0], 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    covariance_term = off_diag.square().sum() / max(feature_dim * (feature_dim - 1), 1)
    return variance_term + covariance_weight * covariance_term
