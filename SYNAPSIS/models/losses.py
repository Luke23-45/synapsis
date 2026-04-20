"""
NTH Loss Functions

Multi-task loss for NTH architecture training:
1. Flow Matching Loss (primary) - Action generation
2. Phase Classification Loss (auxiliary) - Semantic understanding
3. Topological Regularization (auxiliary) - Structure preservation
4. Diversity Loss (auxiliary) - Anchor distribution
"""

from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTHLoss(nn.Module):
    """
    Multi-task loss for NTH architecture training.
    
    Combines:
    - Flow Matching loss for velocity prediction
    - Phase classification for semantic grounding
    - Topological regularization for structure
    - Diversity loss for anchor spread
    
    All weights are configurable for curriculum learning.
    """
    
    def __init__(
        self,
        flow_weight: float = 1.0,
        phase_weight: float = 0.1,
        topo_weight: float = 0.01,
        diversity_weight: float = 0.01,
        use_phase_loss: bool = True,
        use_topo_loss: bool = True,
        use_diversity_loss: bool = True,
    ):
        super().__init__()
        self.flow_weight = flow_weight
        self.phase_weight = phase_weight
        self.topo_weight = topo_weight
        self.diversity_weight = diversity_weight
        self.use_phase_loss = use_phase_loss
        self.use_topo_loss = use_topo_loss
        self.use_diversity_loss = use_diversity_loss
    
    def flow_matching_loss(
        self,
        v_pred: torch.Tensor,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Conditional Flow Matching loss.
        
        Args:
            v_pred: (B, K, D) - Predicted velocity
            x_0: (B, K, D) - Noise (start point)
            x_1: (B, K, D) - Clean actions (end point)
        
        Returns:
            loss: Scalar MSE loss
        """
        v_true = x_1 - x_0
        return F.mse_loss(v_pred, v_true)
    
    def phase_classification_loss(
        self,
        phase_logits: torch.Tensor,
        phase_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Phase classification cross-entropy loss."""
        return F.cross_entropy(phase_logits, phase_labels)
    
    def diversity_loss(
        self,
        anchor_indices: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """
        Encourage temporal diversity in anchor selection.
        Penalizes anchors that are too close together.
        """
        B, K = anchor_indices.shape
        indices_float = anchor_indices.float() / T
        
        # Pairwise distances
        diff = indices_float.unsqueeze(-1) - indices_float.unsqueeze(-2)
        distances = torch.abs(diff)
        
        # Mask diagonal
        mask = 1 - torch.eye(K, device=anchor_indices.device)
        masked_distances = distances + (1 - mask) * 1e6
        min_distances = masked_distances.min(dim=-1).values
        
        # Exponential penalty for small distances
        return torch.exp(-10 * min_distances).mean()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.
        
        Args:
            outputs: Model outputs containing v_pred, phase_logits, etc.
            targets: Ground truth containing actions, noise, phase_labels, etc.
        
        Returns:
            total_loss: Scalar
            loss_dict: Individual losses for logging
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=outputs['v_pred'].device)
        
        # 1. Flow Matching Loss
        flow_loss = self.flow_matching_loss(
            outputs['v_pred'],
            targets['noise'],
            targets['actions'],
        )
        loss_dict['flow_loss'] = flow_loss.item()
        total_loss = total_loss + self.flow_weight * flow_loss
        
        # 2. Phase Loss
        if self.use_phase_loss and 'phase_logits' in outputs and 'phase_labels' in targets:
            phase_loss = self.phase_classification_loss(
                outputs['phase_logits'],
                targets['phase_labels'],
            )
            loss_dict['phase_loss'] = phase_loss.item()
            total_loss = total_loss + self.phase_weight * phase_loss
        
        # 3. Topological Loss
        if self.use_topo_loss and 'topo_loss' in outputs:
            topo_loss = outputs['topo_loss']
            loss_dict['topo_loss'] = topo_loss.item()
            total_loss = total_loss + self.topo_weight * topo_loss
        
        # 4. Diversity Loss
        if self.use_diversity_loss and 'anchor_indices' in outputs:
            T = targets.get('episode_length', 256)
            div_loss = self.diversity_loss(outputs['anchor_indices'], T)
            loss_dict['diversity_loss'] = div_loss.item()
            total_loss = total_loss + self.diversity_weight * div_loss
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


def cosine_temperature_schedule(
    current_epoch: int,
    total_epochs: int,
    temp_start: float = 1.0,
    temp_end: float = 0.1,
) -> float:
    """
    Cosine annealing for Gumbel-Softmax temperature.
    
    Start with high temperature (soft selection) for exploration,
    anneal to low temperature (hard selection) for precision.
    """
    import math
    progress = current_epoch / max(total_epochs, 1)
    progress = min(1.0, max(0.0, progress))
    return temp_end + 0.5 * (temp_start - temp_end) * (1 + math.cos(math.pi * progress))


def linear_warmup_cosine_decay(
    current_step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-6,
    max_lr: float = 1e-4,
) -> float:
    """Linear warmup followed by cosine decay for learning rate."""
    import math
    
    if current_step < warmup_steps:
        return min_lr + (max_lr - min_lr) * current_step / max(warmup_steps, 1)
    else:
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
