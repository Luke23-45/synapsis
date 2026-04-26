"""
Combined loss with auxiliary weight scheduling for Z2 end-to-end training.

Total loss:
  L_total = L_action + α₁(epoch) · L_sparsity + α₂(epoch) · L_topo

Auxiliary weights are ramped linearly from 0:
  - Epochs [0, ramp_start): α = 0 (pure action learning)
  - Epochs [ramp_start, ramp_end): linear ramp to target weight
  - Epochs [ramp_end, ∞): α = target weight

This two-phase strategy lets the model first learn basic action prediction
before auxiliary pressure refines event selection and topology features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from synapse_arch.types import TrainForwardOutput

from .action_loss import action_loss
from .auxiliary_losses import sparsity_loss, topology_reg_loss


@dataclass
class LossConfig:
    """Configuration for the combined loss function."""

    action_weight: float = 1.0
    sparsity_weight: float = 0.01
    topology_reg_weight: float = 0.001
    action_beta: float = 0.5       # Smooth-L1 transition point
    aux_ramp_start: int = 10       # Epoch to begin ramping aux losses
    aux_ramp_end: int = 30         # Epoch where aux losses reach full weight


def _aux_weight_schedule(epoch: int, target_weight: float, ramp_start: int, ramp_end: int) -> float:
    """Compute the auxiliary weight at a given epoch.

    Returns 0.0 before ramp_start, linearly ramps to target_weight
    between ramp_start and ramp_end, then stays at target_weight.
    """
    if epoch < ramp_start:
        return 0.0
    if epoch >= ramp_end:
        return target_weight
    # Linear interpolation
    progress = (epoch - ramp_start) / max(ramp_end - ramp_start, 1)
    return target_weight * progress


def combined_loss(
    output: TrainForwardOutput,
    batch: Dict[str, torch.Tensor],
    epoch: int,
    config: LossConfig,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Compute the combined training loss with auxiliary scheduling.

    Parameters
    ----------
    output : TrainForwardOutput
        Model output from forward_train().
    batch : Dict[str, torch.Tensor]
        Must contain "ground_truth_actions" of shape (B, chunk_size, action_dim).
    epoch : int
        Current training epoch (for auxiliary weight scheduling).
    config : LossConfig
        Loss configuration with weights and scheduling parameters.

    Returns
    -------
    total_loss : torch.Tensor
        Scalar total loss for backpropagation.
    loss_dict : Dict[str, float]
        Breakdown of individual loss components (for logging).
    """
    # Primary loss
    l_action = action_loss(
        output.pred_actions,
        batch["ground_truth_actions"],
        beta=config.action_beta,
    )

    # Auxiliary losses
    l_sparsity = sparsity_loss(output.y_star)
    l_topo = topology_reg_loss(output.topology_token)

    # Scheduled weights
    alpha_sparsity = _aux_weight_schedule(
        epoch, config.sparsity_weight, config.aux_ramp_start, config.aux_ramp_end
    )
    alpha_topo = _aux_weight_schedule(
        epoch, config.topology_reg_weight, config.aux_ramp_start, config.aux_ramp_end
    )

    # Combined
    total = (
        config.action_weight * l_action
        + alpha_sparsity * l_sparsity
        + alpha_topo * l_topo
    )

    loss_dict = {
        "loss_total": total.item(),
        "loss_action": l_action.item(),
        "loss_sparsity": l_sparsity.item(),
        "loss_topo_reg": l_topo.item(),
        "alpha_sparsity": alpha_sparsity,
        "alpha_topo": alpha_topo,
    }

    return total, loss_dict
