"""
Learning rate scheduler: cosine annealing with linear warmup.

During warmup (epochs 0..warmup_epochs-1): LR ramps linearly from 0 to base_lr.
After warmup: LR follows cosine decay from base_lr to min_lr.

This is the standard scheduler for transformer-based models in small-data
regimes and is proven more stable than step-decay for robotics policies.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def create_cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-6,
) -> LambdaLR:
    """Create a cosine-with-warmup LR scheduler.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer whose LR will be scheduled.
    warmup_steps : int
        Number of steps for linear warmup (LR: 0 -> base_lr).
    total_steps : int
        Total number of training steps.
    min_lr : float
        Minimum learning rate at the end of cosine decay.

    Returns
    -------
    LambdaLR
        Scheduler that should be stepped once per batch/step.
    """
    # Extract base LR from the first param group
    base_lr = optimizer.param_groups[0]["lr"]

    # Minimum LR as a fraction of base LR
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup: 0 -> 1 over warmup_steps
            return (step + 1) / max(warmup_steps, 1)
        else:
            # Cosine decay: 1 -> min_lr_ratio over remaining steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = min(progress, 1.0)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

    return LambdaLR(optimizer, lr_lambda)
