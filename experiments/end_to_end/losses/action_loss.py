"""
Primary action prediction loss for Z2 end-to-end training.

Uses Smooth-L1 (Huber) loss for robustness against outlier actions
at contact transitions where the robot undergoes large state changes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def action_loss(
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """Compute Smooth-L1 loss between predicted and ground-truth action chunks.

    Parameters
    ----------
    pred_actions : torch.Tensor
        Shape (B, chunk_size, action_dim). Model predictions.
    gt_actions : torch.Tensor
        Shape (B, chunk_size, action_dim). Ground-truth actions.
    beta : float
        Transition point between L1 and L2 behaviour.
        Below beta: quadratic (good gradient for small errors).
        Above beta: linear (robust to outliers).

    Returns
    -------
    torch.Tensor
        Scalar loss value (mean over all elements).
    """
    return F.smooth_l1_loss(pred_actions, gt_actions, beta=beta)
