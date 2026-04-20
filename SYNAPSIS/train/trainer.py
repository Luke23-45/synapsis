"""
TrainerV3 - Production-Grade Training Infrastructure

SOTA Features:
1. Hydra Configuration Integration
2. Atomic Checkpoint Writes (crash-safe)
3. Epoch Backup with Auto-Cleanup
4. Full State Restoration (optimizer, scheduler, RNG)
5. tqdm Progress Bars with Rich Metrics
6. Optional WandB Logging
7. Seed Control for Reproducibility

Author: Gemini (SYNAPSIS Project)
Version: 3.0
"""

import os
import copy
import time
import signal
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ==============================================================================
# CHECKPOINT MANAGER
# ==============================================================================

class CheckpointManager:
    """
    Manages checkpoint files with atomic writes and automatic cleanup.
    
    Files managed:
    - latest.pt: Always updated every epoch (atomic write)
    - best.pt: Updated when validation loss improves
    - backup_epoch_N.pt: Current epoch backup (previous deleted)
    """
    
    def __init__(self, save_dir: Path, keep_last_n_backups: int = 1):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n_backups = keep_last_n_backups
        self._backup_files = []
    
    def _atomic_save(self, state: dict, path: Path):
        """Save checkpoint atomically to prevent corruption on crash."""
        temp_path = path.with_suffix('.tmp')
        torch.save(state, temp_path)
        # Atomic rename (POSIX-compliant, Windows-compatible via os.replace)
        os.replace(temp_path, path)
    
    def save_latest(self, state: dict):
        """Save latest checkpoint (overwritten every epoch)."""
        path = self.save_dir / 'latest.pt'
        self._atomic_save(state, path)
        return path
    
    def save_best(self, state: dict):
        """Save best model checkpoint."""
        path = self.save_dir / 'best.pt'
        self._atomic_save(state, path)
        return path
    
    def save_backup(self, state: dict, epoch: int):
        """Save epoch backup, delete previous backups beyond keep limit."""
        path = self.save_dir / f'backup_epoch_{epoch:04d}.pt'
        self._atomic_save(state, path)
        self._backup_files.append(path)
        
        # Cleanup old backups
        while len(self._backup_files) > self.keep_last_n_backups:
            old_backup = self._backup_files.pop(0)
            if old_backup.exists():
                old_backup.unlink()
        
        return path
    
    def save_final(self, state: dict):
        """Save final model after training completes."""
        path = self.save_dir / 'final.pt'
        self._atomic_save(state, path)
        return path
    
    def get_resume_path(self) -> Optional[Path]:
        """Get the best checkpoint to resume from."""
        latest = self.save_dir / 'latest.pt'
        if latest.exists():
            return latest
        
        # Fallback to most recent backup
        backups = sorted(self.save_dir.glob('backup_epoch_*.pt'))
        if backups:
            return backups[-1]
        
        return None


# ==============================================================================
# TRAINER V3
# ==============================================================================

class TrainerV3:
    """
    Production-grade trainer for NTH Diffusion Planner.
    
    Features:
    - Hydra-compatible (accepts OmegaConf DictConfig)
    - Atomic checkpoint writes
    - Epoch backup with auto-cleanup
    - Full state restoration
    - tqdm progress bars
    - Optional WandB logging
    - Seed control
    - Graceful shutdown on SIGTERM/SIGINT
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Any,  # OmegaConf DictConfig or NTHConfig dataclass
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda',
        save_dir: str = 'checkpoints',
        use_wandb: bool = False,
        wandb_project: str = 'SYNAPSIS',
        seed: int = 42,
    ):
        # Config handling (support both OmegaConf and dataclass)
        self.config = config
        self._extract_config_values(config)
        
        # Device setup
        self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
        if self.device != device:
            print(f"[TrainerV3] Warning: {device} not available, using {self.device}")
        
        # Model
        self.model = model.to(self.device)
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Seed for reproducibility
        self.seed = seed
        self._set_seed(seed)
        
        # Optimizer with weight decay groups
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler() if self.use_amp and self.device == 'cuda' else None
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(save_dir, keep_last_n_backups=1)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_complete = False
        
        # Signal handling for graceful shutdown
        self._setup_signal_handlers()
        self._shutdown_requested = False
        
        # WandB
        self.use_wandb = use_wandb and HAS_WANDB
        if self.use_wandb:
            self._init_wandb(wandb_project)
    
    def _extract_config_values(self, config):
        """Extract training hyperparameters from config."""
        # Support both attribute access (dataclass) and dict access (OmegaConf)
        def get(key, default):
            if hasattr(config, key):
                return getattr(config, key)
            elif hasattr(config, 'training') and hasattr(config.training, key):
                return getattr(config.training, key)
            elif hasattr(config, 'get'):
                return config.get(key, default)
            return default
        
        self.learning_rate = get('learning_rate', 1e-4)
        self.weight_decay = get('weight_decay', 0.01)
        self.max_grad_norm = get('max_grad_norm', 1.0)
        self.warmup_steps = get('warmup_steps', 1000)
        self.max_epochs = get('max_epochs', 100)
        self.use_amp = get('use_amp', True)
        self.val_check_interval = get('val_check_interval', 1)
        self.log_interval = get('log_interval', 10)
        self.gradient_accumulation_steps = get('gradient_accumulation_steps', 1)
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Note: For full determinism, also set torch.backends.cudnn.deterministic = True
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay groups."""
        # Separate params: no decay for bias and LayerNorm
        no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight']
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
            },
        ]
        return torch.optim.AdamW(param_groups, lr=self.learning_rate, betas=(0.9, 0.999))
    
    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler with warmup + cosine decay."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(1, self.max_epochs * len(self.train_loader) - self.warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown."""
        def handler(signum, frame):
            print(f"\n[TrainerV3] Received signal {signum}, saving checkpoint and shutting down...")
            self._shutdown_requested = True
        
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    
    def _init_wandb(self, project: str):
        """Initialize WandB logging."""
        wandb.init(
            project=project,
            config=dict(self.config) if hasattr(self.config, '__iter__') else {},
            resume='allow',
        )
        wandb.watch(self.model, log='gradients', log_freq=100)
    
    def get_checkpoint_state(self) -> dict:
        """Create full training state for checkpointing."""
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
            },
            'config': self.config,
        }
        
        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()
        
        if torch.cuda.is_available():
            state['rng_state']['cuda'] = torch.cuda.get_rng_state_all()
        
        return state
    
    def load_checkpoint(self, path: Optional[str] = None):
        """Load checkpoint and restore full training state."""
        if path is None:
            path = self.checkpoint_manager.get_resume_path()
        
        if path is None:
            print("[TrainerV3] No checkpoint found, starting fresh.")
            return False
        
        print(f"[TrainerV3] Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore scaler
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore RNG states
        if 'rng_state' in checkpoint:
            rng = checkpoint['rng_state']
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'])
            if 'cuda' in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng['cuda'])
        
        # Restore training state
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"[TrainerV3] Resumed from epoch {self.epoch}, step {self.global_step}")
        return True
    
    def train_step(self, batch: dict) -> dict:
        """Execute single training step."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass with AMP
        with torch.amp.autocast(device_type='cuda', enabled=self.use_amp and self.device == 'cuda'):
            outputs = self.model(batch)
            loss, loss_dict = self.model.compute_loss(outputs, batch)
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Step optimizer every N batches (gradient accumulation)
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            loss_dict['grad_norm'] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            loss_dict['lr'] = self.scheduler.get_last_lr()[0]
        
        self.global_step += 1
        return loss_dict
    
    def train_epoch(self) -> dict:
        """Train for one epoch with progress bar."""
        epoch_losses = defaultdict(list)
        
        # Sync progress to model for curriculum/annealing
        if hasattr(self.model, 'current_epoch'):
            self.model.current_epoch = self.epoch
        
        # Setup progress bar
        if HAS_TQDM:
            pbar = tqdm(
                self.train_loader, 
                desc=f"Epoch {self.epoch + 1}/{self.max_epochs}",
                leave=True,
                dynamic_ncols=True
            )
        else:
            pbar = self.train_loader
        
        for batch_idx, batch in enumerate(pbar):
            # Check for shutdown request
            if self._shutdown_requested:
                print("[TrainerV3] Shutdown requested, saving and exiting...")
                self._save_and_exit()
                return {}
            
            loss_dict = self.train_step(batch)
            
            # Accumulate losses
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    epoch_losses[k].append(v)
            
            # Update progress bar
            if HAS_TQDM and batch_idx % self.log_interval == 0:
                postfix = {
                    'loss': f"{loss_dict.get('total_loss', 0):.4f}",
                    'lr': f"{loss_dict.get('lr', 0):.2e}",
                }
                pbar.set_postfix(postfix)
            
            # WandB logging
            if self.use_wandb and self.global_step % self.log_interval == 0:
                wandb.log({f"train/{k}": v for k, v in loss_dict.items()}, step=self.global_step)
        
        self.epoch += 1
        
        # Return epoch averages
        return {k: sum(v) / len(v) for k, v in epoch_losses.items() if v}
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation loop."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_losses = defaultdict(list)
        
        if HAS_TQDM:
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        else:
            pbar = self.val_loader
        
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            outputs = self.model(batch)
            _, loss_dict = self.model.compute_loss(outputs, batch)
            
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    val_losses[k].append(v)
        
        return {k: sum(v) / len(v) for k, v in val_losses.items() if v}
    
    def _save_and_exit(self):
        """Save checkpoint and exit gracefully."""
        state = self.get_checkpoint_state()
        self.checkpoint_manager.save_latest(state)
        print("[TrainerV3] Emergency checkpoint saved. Exiting.")
        exit(0)
    
    def fit(self, resume: bool = True):
        """Main training loop."""
        # Resume if requested
        if resume:
            self.load_checkpoint()
        
        print("=" * 60)
        print(f"[TrainerV3] Starting training")
        print(f"  - Epochs: {self.epoch} -> {self.max_epochs}")
        print(f"  - Device: {self.device}")
        print(f"  - AMP: {self.use_amp}")
        print(f"  - WandB: {self.use_wandb}")
        print("=" * 60)
        
        for epoch in range(self.epoch, self.max_epochs):
            start_time = time.time()
            
            # Train
            train_losses = self.train_epoch()
            if not train_losses:  # Shutdown requested
                return
            
            elapsed = time.time() - start_time
            
            # Validate
            val_losses = {}
            if self.val_loader is not None and (epoch + 1) % self.val_check_interval == 0:
                val_losses = self.validate()
            
            # Log epoch summary
            train_loss = train_losses.get('total_loss', 0)
            val_loss = val_losses.get('total_loss', train_loss)
            
            print(
                f"Epoch {self.epoch:3d}/{self.max_epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"Time: {elapsed:.1f}s"
            )
            
            # WandB epoch logging
            if self.use_wandb:
                wandb.log({
                    "epoch": self.epoch,
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss": val_loss,
                }, step=self.global_step)
            
            # Checkpoint management
            state = self.get_checkpoint_state()
            
            # Always save latest
            self.checkpoint_manager.save_latest(state)
            
            # Save backup (auto-cleanup of previous)
            self.checkpoint_manager.save_backup(state, self.epoch)
            
            # Save best if improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.checkpoint_manager.save_best(state)
                print(f"  -> New best model saved! (val_loss: {val_loss:.4f})")
        
        # Training complete
        self.training_complete = True
        state = self.get_checkpoint_state()
        self.checkpoint_manager.save_final(state)
        
        print("=" * 60)
        print("[TrainerV3] Training complete!")
        print(f"  - Best val loss: {self.best_val_loss:.4f}")
        print(f"  - Final epoch: {self.epoch}")
        print("=" * 60)
        
        if self.use_wandb:
            wandb.finish()
