"""
NTH Utility Functions

Common utilities for the NTH architecture.
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_parameter_groups(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> list:
    """
    Get parameter groups with selective weight decay.
    
    No weight decay on:
    - Biases
    - LayerNorm parameters
    - Embeddings
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'bias' in name or 'norm' in name.lower() or 'embed' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {'params': decay_params, 'weight_decay': weight_decay, 'lr': lr},
        {'params': no_decay_params, 'weight_decay': 0.0, 'lr': lr},
    ]


def initialize_weights(model: nn.Module):
    """
    Initialize model weights following best practices.
    
    - Linear: Xavier uniform
    - Embeddings: Normal(0, 0.02)
    - LayerNorm: bias=0, weight=1
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)


def create_dummy_batch(
    batch_size: int = 4,
    image_size: int = 224,
    proprio_dim: int = 22,
    proprio_horizon: int = 256,
    action_dim: int = 8,
    action_chunk_size: int = 8,
    device: str = 'cpu',
) -> Dict[str, torch.Tensor]:
    """Create dummy batch for testing."""
    return {
        'image': torch.randn(batch_size, 3, image_size, image_size, device=device),
        'proprio': torch.randn(batch_size, proprio_dim, device=device),
        'proprio_history': torch.randn(batch_size, proprio_horizon, proprio_dim, device=device),
        'action_chunk': torch.randn(batch_size, action_chunk_size, action_dim, device=device),
        'phase_labels': torch.randint(0, 5, (batch_size,), device=device),
    }
