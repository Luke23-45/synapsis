"""
NTH-Attention Training Script with Hydra Configuration

This is the Hydra-enabled entry point that works alongside the existing config.py.
It demonstrates how to use YAML configs with CLI overrides.

Usage:
    # Default config
    python train_hydra.py
    
    # Switch to small model
    python train_hydra.py model=small
    
    # Override learning rate
    python train_hydra.py training.learning_rate=3e-4
    
    # Combined
    python train_hydra.py model=large training=fast training.batch_size=64
"""

import logging
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn

# Add parent directory to path for imports when running from SYNAPSIS/
_current_dir = Path(__file__).parent
if str(_current_dir.parent) not in sys.path:
    sys.path.insert(0, str(_current_dir.parent))

# Register configs before Hydra loads
from SYNAPSIS.config_store import register_configs, get_config_from_yaml
register_configs()

# Import model and training utilities
from SYNAPSIS.config import NTHConfig
from SYNAPSIS.models.planner import NTHDiffusionPlanner

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra-enabled training entry point.
    
    Args:
        cfg: OmegaConf DictConfig loaded from YAML + CLI overrides
    """
    # Print resolved config
    log.info("=" * 60)
    log.info("SYNAPSIS Training with Hydra Configuration")
    log.info("=" * 60)
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
    # Convert to NTHConfig dataclass for type safety
    try:
        config: NTHConfig = get_config_from_yaml(cfg)
        log.info(f"Config validated successfully")
        log.info(f"   - d_model: {config.d_model}")
        log.info(f"   - num_layers: {config.num_layers}")
        log.info(f"   - learning_rate: {config.learning_rate}")
    except Exception as e:
        log.error(f"Config validation failed: {e}")
        raise
    
    # Device setup
    device = config.get_device()
    log.info(f"Using device: {device}")
    
    # Model instantiation
    log.info("Instantiating NTHDiffusionPlanner...")
    model = NTHDiffusionPlanner(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total parameters: {total_params / 1e6:.2f}M")
    log.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # 1. Dataset Initialization
    from SYNAPSIS.datasets.expert_dataset import ExpertTrajectoryDataset
    from SYNAPSIS.datasets.data import NTHDataset
    from torch.utils.data import DataLoader
    
    # --- TRAINING DATASET ---
    log.info(f"Loading TRAINING dataset from: {config.train_path}...")
    train_base = ExpertTrajectoryDataset(
        demo_path=config.train_path,
        observation_horizon=1,
        action_horizon=config.action_chunk_size
    )
    train_dataset = NTHDataset(
        base_dataset=train_base,
        proprio_horizon=config.proprio_horizon,
        action_chunk_size=config.action_chunk_size,
        image_size=config.image_size,
        use_wrist_camera=True,
        use_awr=getattr(config, 'use_awr', False)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )

    # --- VALIDATION DATASET (Optional) ---
    val_loader = None
    if config.val_path:
        log.info(f"Loading VALIDATION dataset from: {config.val_path}...")
        try:
            val_base = ExpertTrajectoryDataset(
                demo_path=config.val_path,
                observation_horizon=1,
                action_horizon=config.action_chunk_size
            )
            val_dataset = NTHDataset(
                base_dataset=val_base,
                proprio_horizon=config.proprio_horizon,
                action_chunk_size=config.action_chunk_size,
                image_size=config.image_size,
                use_wrist_camera=True,
                use_awr=getattr(config, 'use_awr', False)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False, # No shuffle for validation
                num_workers=2, # Fewer workers for val
                pin_memory=True
            )
        except Exception as e:
            log.warning(f"Could not initialize validation dataset: {e}. Proceeding without validation.")

    # 3. Trainer V3 Infrastructure
    log.info("Starting TrainerV3 Engine...")
    from SYNAPSIS.train.trainer import TrainerV3
    trainer = TrainerV3(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader, # Now passing the val_loader
        device=str(device),
        save_dir=config.save_dir,
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        seed=config.seed
    )
    
    # 4. Training Execution (With Auto-Resume)
    log.info("=" * 60)
    log.info("INITIALIZING TRAINING LOOP")
    log.info("=" * 60)
    trainer.fit(resume=True)
    
    log.info("=" * 60)
    log.info("TRAINING SESSION COMPLETED SUCCESSFULLY")
    log.info("=" * 60)
    
    return None


if __name__ == "__main__":
    main()
