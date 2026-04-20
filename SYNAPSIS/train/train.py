"""
NTH-Attention Training Script v2.0

SOTA training loop with:
- Mixed precision (AMP)
- Gradient clipping
- Temperature annealing
- Proper learning rate scheduling
- Checkpoint saving/loading
"""

import argparse
from collections import defaultdict
from pathlib import Path
import time
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nth import (
    NTHConfig,
    NTHConfigBase,
    NTHDiffusionPlanner,
    create_synthetic_dataloader,
    cosine_temperature_schedule,
    linear_warmup_cosine_decay,
)
from nth.utils import get_parameter_groups, count_parameters


class Trainer:
    """
    SOTA Trainer for NTH Diffusion Planner.
    
    Features:
    - Mixed precision (AMP) training
    - Gradient clipping
    - Temperature annealing for curriculum learning
    - Learning rate warmup + cosine decay
    - Comprehensive logging
    - Checkpoint management
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: NTHConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_dir: str = 'checkpoints',
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer with weight decay groups
        param_groups = get_parameter_groups(
            model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler() if config.use_amp and device == 'cuda' else None
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
    
    def train_step(self, batch: dict) -> dict:
        """Single training step with gradient accumulation support."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass with AMP
        with torch.amp.autocast(device_type='cuda', enabled=self.config.use_amp and self.device == 'cuda'):
            outputs = self.model(batch)
            loss, loss_dict = self.model.compute_loss(outputs, batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
        
        loss_dict['grad_norm'] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        self.global_step += 1
        
        return loss_dict
    
    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch."""
        epoch_losses = defaultdict(list)
        
        for batch_idx, batch in enumerate(dataloader):
            loss_dict = self.train_step(batch)
            
            for k, v in loss_dict.items():
                epoch_losses[k].append(v)
            
            # Update learning rate
            lr = linear_warmup_cosine_decay(
                self.global_step,
                self.config.warmup_steps,
                self.config.warmup_steps + len(dataloader) * self.config.temp_anneal_epochs,
                min_lr=self.config.learning_rate * 0.01,
                max_lr=self.config.learning_rate,
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        # Temperature annealing
        new_temp = cosine_temperature_schedule(
            self.epoch,
            self.config.temp_anneal_epochs,
            self.config.gumbel_temperature,
            self.config.temp_end,
        )
        self.model.anneal_temperature(new_temp)
        
        self.epoch += 1
        
        return {k: sum(v) / len(v) for k, v in epoch_losses.items()}
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Validation loop."""
        self.model.eval()
        val_losses = defaultdict(list)
        
        for batch in dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            outputs = self.model(batch)
            _, loss_dict = self.model.compute_loss(outputs, batch)
            
            for k, v in loss_dict.items():
                val_losses[k].append(v)
        
        return {k: sum(v) / len(v) for k, v in val_losses.items()}
    
    def save_checkpoint(self, filename: str = None, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if filename is None:
            filename = f'checkpoint_epoch_{self.epoch}.pt'
        
        path = self.save_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Resumed from epoch {self.epoch}, step {self.global_step}")


def main():
    parser = argparse.ArgumentParser(description='Train NTH Diffusion Planner v2.0')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--scale', type=str, default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--num_samples', type=int, default=1000, help='Synthetic samples')
    args = parser.parse_args()
    
    # Verify device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Config
    from nth import get_config
    config = get_config(args.scale)
    config.learning_rate = args.lr
    
    print(f"=" * 60)
    print(f"NTH-Attention Training v2.0")
    print(f"=" * 60)
    print(f"Scale: {args.scale}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"=" * 60)
    
    # Model
    model = NTHDiffusionPlanner(config, validate_inputs=True, validate_outputs=True)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Data (synthetic for testing)
    train_loader = create_synthetic_dataloader(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        proprio_dim=config.proprio_dim,
        action_dim=config.action_dim,
        proprio_horizon=config.proprio_horizon,
        action_chunk_size=config.action_chunk_size,
        image_size=config.image_size,
        num_phases=config.num_phases,
    )
    print(f"Training samples: {args.num_samples}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Trainer
    trainer = Trainer(model, config, device=device, save_dir=args.save_dir)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    print(f"=" * 60)
    print("Starting training...")
    print(f"=" * 60)
    
    # Training loop
    for epoch in range(trainer.epoch, args.epochs):
        start = time.time()
        
        # Train
        train_losses = trainer.train_epoch(train_loader)
        elapsed = time.time() - start
        
        # Log
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Loss: {train_losses['total_loss']:.4f} | "
            f"Flow: {train_losses['flow_loss']:.4f} | "
            f"Grad: {train_losses['grad_norm']:.2f} | "
            f"Time: {elapsed:.1f}s"
        )
        
        # Checkpoint
        is_best = train_losses['total_loss'] < trainer.best_loss
        if is_best:
            trainer.best_loss = train_losses['total_loss']
        
        if (epoch + 1) % 10 == 0 or is_best:
            trainer.save_checkpoint(is_best=is_best)
    
    # Final save
    trainer.save_checkpoint('final_model.pt')
    print(f"=" * 60)
    print("Training complete!")
    print(f"Best loss: {trainer.best_loss:.4f}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
