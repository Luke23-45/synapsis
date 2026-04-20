#!/usr/bin/env python3
"""
train_v3.py - Hydra-Integrated Training Entry Point

Production-grade training script that uses:
1. Hydra YAML configuration (no CLI argparse)
2. TrainerV3 with robust checkpointing
3. Automatic resume from latest checkpoint
4. Full integration with SYNAPSIS V3 model

Usage:
    # Default training
    python train_v3.py
    
    # Override config
    python train_v3.py model=small training.learning_rate=3e-4
    
    # Resume from checkpoint
    python train_v3.py +resume=true
    
    # Change output directory
    python train_v3.py hydra.run.dir=./outputs/experiment_01

Author: Gemini (SYNAPSIS Project)
"""

import sys
from pathlib import Path

# Add parent path for imports (handle running from different directories)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Register configs before hydra.main
from SYNAPSIS.config_store import register_configs
register_configs()


def setup_device(cfg: DictConfig) -> str:
    """Determine device from config, with fallback."""
    device = cfg.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("[train_v3] CUDA not available, falling back to CPU")
        return 'cpu'
    return device


def create_model(cfg: DictConfig, device: str):
    """Create NTHDiffusionPlanner from config."""
    from SYNAPSIS.models.planner import NTHDiffusionPlanner
    from SYNAPSIS.config import NTHConfig
    
    # Convert OmegaConf to dataclass
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    
    # Build NTHConfig
    config = NTHConfig(
        # Model architecture
        d_model=model_cfg.get('d_model', 768),
        num_heads=model_cfg.get('num_heads', 12),
        num_layers=model_cfg.get('num_layers', 12),
        num_kv_heads=model_cfg.get('num_kv_heads', None),
        
        # Vision
        vision_backbone=model_cfg.get('vision_backbone', 'google/siglip-base-patch16-224'),
        unfreeze_last_n=model_cfg.get('unfreeze_last_n', 3),
        use_gradient_checkpointing=model_cfg.get('use_gradient_checkpointing', True),
        
        # Input dimensions
        proprio_dim=cfg.get('proprio_dim', 22),
        action_dim=cfg.get('action_dim', 8),
        image_size=cfg.get('image_size', 224),
        
        # Temporal
        proprio_horizon=cfg.sampler.get('proprio_horizon', 32),
        num_anchors=cfg.sampler.get('num_anchors', 16),
        
        # Action head
        action_chunk_size=model_cfg.action_head.get('chunk_size', 8),
        action_head_layers=model_cfg.action_head.get('num_layers', 4),
        action_head_heads=model_cfg.action_head.get('num_heads', 8),
        
        # Training (from training config)
        learning_rate=cfg.training.get('learning_rate', 1e-4),
        weight_decay=cfg.training.get('weight_decay', 0.01),
        dropout=model_cfg.get('dropout', 0.1),
        max_grad_norm=cfg.training.get('max_grad_norm', 1.0),
        
        # Losses
        flow_loss_weight=cfg.training.get('flow_loss_weight', 1.0),
        phase_loss_weight=cfg.training.get('phase_loss_weight', 0.1),
        diversity_loss_weight=cfg.training.get('diversity_loss_weight', 0.01),
        topo_loss_weight=cfg.training.get('topo_loss_weight', 0.01),
        consistency_loss_weight=cfg.training.get('consistency_loss_weight', 0.1),
        use_consistency_loss=cfg.training.get('use_consistency_loss', True),
        
        # Misc
        use_amp=cfg.training.get('use_amp', True),
        num_phases=cfg.get('num_phases', 5),
        use_phase_prediction=cfg.get('use_phase_prediction', True),
        device=device,
    )
    
    model = NTHDiffusionPlanner(
        config,
        validate_inputs=True,
        validate_outputs=True,
    )
    
    return model, config


def create_dataloaders(cfg: DictConfig, config):
    """Create train and validation dataloaders."""
    # For now, use synthetic data - replace with real dataset later
    from SYNAPSIS.datasets.data import create_synthetic_dataloader
    
    batch_size = cfg.training.get('batch_size', 32)
    num_samples = cfg.training.get('num_samples', 1000)
    
    train_loader = create_synthetic_dataloader(
        num_samples=num_samples,
        batch_size=batch_size,
        proprio_dim=config.proprio_dim,
        action_dim=config.action_dim,
        proprio_horizon=config.proprio_horizon,
        action_chunk_size=config.action_chunk_size,
        image_size=config.image_size,
        num_phases=config.num_phases,
    )
    
    # Optional validation loader
    val_loader = None
    if cfg.training.get('val_samples', 0) > 0:
        val_loader = create_synthetic_dataloader(
            num_samples=cfg.training.val_samples,
            batch_size=batch_size,
            proprio_dim=config.proprio_dim,
            action_dim=config.action_dim,
            proprio_horizon=config.proprio_horizon,
            action_chunk_size=config.action_chunk_size,
            image_size=config.image_size,
            num_phases=config.num_phases,
        )
    
    return train_loader, val_loader


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training entry point."""
    # Print resolved config
    print("=" * 60)
    print("SYNAPSIS Training V3.0 (Hydra-Integrated)")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Setup
    device = setup_device(cfg)
    print(f"[train_v3] Device: {device}")
    
    # Create model
    print("[train_v3] Creating model...")
    model, config = create_model(cfg, device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_v3] Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Create dataloaders
    print("[train_v3] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(cfg, config)
    print(f"[train_v3] Train batches: {len(train_loader)}")
    
    # Create trainer
    from SYNAPSIS.train.trainer import TrainerV3
    
    trainer = TrainerV3(
        model=model,
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=cfg.training.get('save_dir', 'checkpoints'),
        use_wandb=cfg.training.get('use_wandb', False),
        wandb_project=cfg.training.get('wandb_project', 'SYNAPSIS'),
        seed=cfg.training.get('seed', 42),
    )
    
    # Resume from checkpoint if requested
    resume = cfg.get('resume', True)
    
    # Run training
    print("[train_v3] Starting training...")
    trainer.fit(resume=resume)
    
    print("[train_v3] Done!")


if __name__ == "__main__":
    main()
