"""
NTH-Attention Configuration v2.0 (SOTA)

Defines all hyperparameters for the NTH architecture with SOTA defaults.
Includes configurations for Small, Base, and Large model scales.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
import torch


@dataclass
class NTHConfig:
    """
    Configuration for the NTH-Attention Architecture v2.0.
    
    SOTA defaults based on:
    - Octo-Base (93M params)
    - Pi0 action head design
    - Diffusion Transformer Policy (DiT-style)
    
    Sections:
    1. Input Dimensions
    2. Vision Encoder (SigLIP)
    3. Temporal Sampler (Module 1)
    4. Geometric Projector (Module 2)
    5. NTH-Attention Transformer (Module 3)
    6. Flow Matching Action Head
    7. Training
    8. Data Pipeline
    """
    
    # =========================================================================
    # 1. INPUT DIMENSIONS
    # =========================================================================
    proprio_dim: int = 22           # Proprioception dimension
    action_dim: int = 8             # Action dimension (7 pose + 1 gripper)
    image_size: int = 224           # Input image size (H=W)
    image_channels: int = 3         # RGB
    
    # =========================================================================
    # 2. VISION ENCODER (SOTA: SigLIP)
    # =========================================================================
    vision_backbone: str = "google/siglip-base-patch16-224"
    vision_hidden_dim: int = 768    # SigLIP-base hidden dimension
    vision_patch_size: int = 16     # Patch size for ViT
    freeze_vision: bool = True      # Freeze most vision layers
    unfreeze_last_n: int = 3        # Unfreeze last N encoder layers
    use_gradient_checkpointing: bool = True  # Memory optimization
    num_views: int = 3              # primary, wrist, goal
    use_siglip: bool = True         # Use SigLIP (False = fallback encoder)
    
    # =========================================================================
    # 3. TEMPORAL SAMPLER (Module 1)
    # =========================================================================
    num_anchors: int = 16           # K: Number of sparse anchors
    proprio_horizon: int = 256      # T: History length for anchoring
    
    # V1 parameters (kept for backward compatibility)
    gate_hidden_dim: int = 128      # Hidden dim for gating MLP
    gumbel_temperature: float = 1.0 # Initial Gumbel-Softmax temperature
    use_dftopk: bool = False        # Use DFTopK instead of Gumbel-Softmax
    compute_durations: bool = True  # Compute time spans between anchors
    
    # V2 parameters (new SOTA architecture)
    tcn_channels: List[int] = field(default_factory=lambda: [64, 64, 64])  # TCN channel sizes
    hysteresis_tau_init: float = 0.5  # Initial tau for adaptive hysteresis
    
    # =========================================================================
    # 4. GEOMETRIC PROJECTOR (Module 2)
    # =========================================================================
    ssm_normalize: bool = True      # Normalize SSM to [0, 1]
    distance_metric: str = "euclidean"  # 'euclidean', 'cosine', 'learned'
    use_learned_metric: bool = False    # Learn projection before distance
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    return_spatial_bias: bool = True    # Return B_geo for attention
    use_topo_regularization: bool = True  # Add topological loss terms
    
    # =========================================================================
    # 5. NTH-ATTENTION TRANSFORMER (Module 3) - SOTA BASE SCALE
    # =========================================================================
    d_model: int = 768              # Transformer hidden dimension (Base)
    num_heads: int = 12             # Number of attention heads (Base)
    num_layers: int = 12            # Number of transformer layers (Base)
    dropout: float = 0.1            # Dropout rate
    max_seq_len: int = 512          # Maximum sequence length
    
    # V1 parameters (kept for backward compatibility)
    ffn_ratio: float = 4.0          # FFN expansion ratio
    lambda_geo_init: float = 0.5    # Initial geometric bias weight
    use_cross_attention: bool = True  # Cross-attend to vision context
    
    # V2 parameters (SOTA: GQA + DropPath)
    num_kv_heads: Optional[int] = None  # GQA: key-value heads (None = standard MHA)
    drop_path_rate: float = 0.1     # Stochastic depth max rate

    
    # =========================================================================
    # 6. FLOW MATCHING ACTION HEAD (SOTA: Pi0-style)
    # =========================================================================
    action_chunk_size: int = 8      # K_act: Number of future actions
    action_head_layers: int = 4     # Layers in action denoiser
    action_head_heads: int = 8      # Heads in action denoiser
    flow_steps_train: int = 1       # Steps during training (always 1 for FM)
    flow_steps_inference: int = 10  # Steps during inference
    use_separate_heads: bool = True # Separate heads for pose/gripper
    
    # =========================================================================
    # 7. TRAINING
    # =========================================================================
    learning_rate: float = 1e-4     # AdamW learning rate
    weight_decay: float = 0.01      # AdamW weight decay
    warmup_steps: int = 1000        # LR warmup steps
    max_grad_norm: float = 1.0      # Gradient clipping
    use_amp: bool = True            # Mixed precision training
    
    # Batch and epochs (for Hydra compatibility)
    batch_size: int = 32            # Training batch size
    max_epochs: int = 100           # Maximum training epochs
    val_check_interval: float = 0.25  # Validation frequency
    gradient_accumulation_steps: int = 1  # Gradient accumulation
    
    # Loss weights
    flow_loss_weight: float = 1.0   # Flow matching loss weight
    phase_loss_weight: float = 0.1  # Phase classification loss weight
    topo_loss_weight: float = 0.01  # Topological regularization weight
    diversity_loss_weight: float = 0.01  # Anchor diversity weight
    consistency_loss_weight: float = 0.1  # Manifold consistency loss weight
    use_consistency_loss: bool = True  # Enable manifold consistency
    
    # AWR (Advantage-Weighted Regression)
    use_awr: bool = False             # Toggle Advantage-Weighted Regression
    awr_beta: float = 1.0             # Initial temperature (w = exp(A/beta))
    awr_beta_end: float = 0.05        # Final temperature (lower = more picky)
    awr_anneal_epochs: int = 50       # Epochs to linear anneal beta
    awr_max_weight: float = 10.0      # Clipping limit for weights to ensure stability
    
    # Curriculum learning
    temp_anneal_epochs: int = 50    # Epochs to anneal temperature
    temp_end: float = 0.1           # Final temperature
    lambda_anneal_epochs: int = 50  # Epochs to anneal lambda_geo
    
    # =========================================================================
    # 8. AUXILIARY TASKS
    # =========================================================================
    num_phases: int = 5             # Number of task phases
    use_phase_prediction: bool = True  # Enable phase auxiliary loss
    
    # =========================================================================
    # 9. DEVICE AND DTYPE
    # =========================================================================
    # = [PRODUCTION PATHS] ====================================================
    train_path: str = ""           # Path to training LMDB dataset
    val_path: str = ""             # Path to validation LMDB dataset
    urdf_path: str = ""             # Path to robot URDF
    xml_path: str = ""              # Path to MuJoCo XML
    # =========================================================================

    device: str = "cuda"            # Target device
    dtype: str = "float32"          # Default dtype (float16 for AMP)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Comprehensive validation of all parameters."""
        # Dimensional checks
        assert self.proprio_dim > 0, f"proprio_dim must be positive, got {self.proprio_dim}"
        assert self.action_dim > 0, f"action_dim must be positive, got {self.action_dim}"
        assert self.image_size in [224, 256, 384, 512], \
            f"image_size must be 224/256/384/512, got {self.image_size}"
        assert self.image_channels == 3, f"Only RGB supported, got {self.image_channels} channels"
        
        # Architecture checks
        assert self.d_model > 0, f"d_model must be positive, got {self.d_model}"
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        assert self.num_layers > 0, f"num_layers must be positive, got {self.num_layers}"
        assert self.num_anchors > 0, f"num_anchors must be positive, got {self.num_anchors}"
        assert self.proprio_horizon >= self.num_anchors, \
            f"proprio_horizon ({self.proprio_horizon}) must be >= num_anchors ({self.num_anchors})"
        
        # Training checks
        assert 0 < self.gumbel_temperature <= 10, \
            f"gumbel_temperature should be in (0, 10], got {self.gumbel_temperature}"
        assert 0 <= self.lambda_geo_init <= 2, \
            f"lambda_geo_init should be in [0, 2], got {self.lambda_geo_init}"
        assert self.learning_rate > 0, f"learning_rate must be positive, got {self.learning_rate}"
        assert self.max_grad_norm > 0, f"max_grad_norm must be positive, got {self.max_grad_norm}"
        
        # Action head checks
        assert self.action_chunk_size > 0, \
            f"action_chunk_size must be positive, got {self.action_chunk_size}"
        assert self.flow_steps_inference >= 1, \
            f"flow_steps_inference must be >= 1, got {self.flow_steps_inference}"
        
        # Distance metric check
        assert self.distance_metric in ["euclidean", "cosine", "manhattan", "learned"], \
            f"Invalid distance_metric: {self.distance_metric}"
        
        # CNN channels check
        assert len(self.cnn_channels) >= 2, \
            f"cnn_channels must have at least 2 elements, got {len(self.cnn_channels)}"
        assert all(c > 0 for c in self.cnn_channels), \
            f"All cnn_channels must be positive, got {self.cnn_channels}"
    
    def get_device(self) -> torch.device:
        """Get torch device from config."""
        if self.device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.device)
    
    def get_dtype(self) -> torch.dtype:
        """Get torch dtype from config."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.float32)
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.num_heads
    
    @property
    def num_vision_patches(self) -> int:
        """Number of patches per image."""
        return (self.image_size // self.vision_patch_size) ** 2
    
    def estimate_parameters(self) -> int:
        """Rough estimate of total model parameters."""
        # Vision encoder (SigLIP-base = ~86M, but mostly frozen)
        vision_trainable = 3 * 12 * (768 ** 2) * 4  # 3 unfrozen layers
        
        # Temporal sampler
        temporal = (2 * self.proprio_dim * self.gate_hidden_dim) + \
                   (self.gate_hidden_dim ** 2) + self.gate_hidden_dim
        
        # Geometric projector (CNN)
        cnn_params = sum(
            self.cnn_channels[i] * self.cnn_channels[i+1] * 9 
            for i in range(len(self.cnn_channels) - 1)
        ) + self.cnn_channels[-1] * self.d_model
        
        # Transformer
        layer_params = 4 * (self.d_model ** 2) + 2 * (self.d_model * 4 * self.d_model)
        transformer = self.num_layers * layer_params
        
        # Action head
        action_head = self.action_head_layers * (4 * (self.d_model ** 2))
        
        return int(vision_trainable + temporal + cnn_params + transformer + action_head)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

@dataclass
class NTHConfigSmall(NTHConfig):
    """
    Small configuration (~27M trainable parameters).
    Matches Octo-Small scale for fast experimentation.
    """
    d_model: int = 384
    num_heads: int = 6
    num_layers: int = 8
    vision_hidden_dim: int = 384
    cnn_channels: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    action_head_layers: int = 3
    action_head_heads: int = 6


@dataclass
class NTHConfigBase(NTHConfig):
    """
    Base configuration (~93M trainable parameters).
    Matches Octo-Base scale - recommended for most experiments.
    """
    # Inherits all defaults from NTHConfig
    pass


@dataclass
class NTHConfigLarge(NTHConfig):
    """
    Large configuration (~300M+ trainable parameters).
    For scaling experiments with more compute.
    """
    d_model: int = 1024
    num_heads: int = 16
    num_layers: int = 16
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    action_head_layers: int = 6
    action_head_heads: int = 16
    use_gradient_checkpointing: bool = True


def get_config(scale: Literal["small", "base", "large"] = "base") -> NTHConfig:
    """
    Factory function to get configuration by scale.
    
    Args:
        scale: One of 'small', 'base', 'large'
    
    Returns:
        NTHConfig instance
    """
    configs = {
        "small": NTHConfigSmall,
        "base": NTHConfigBase,
        "large": NTHConfigLarge,
    }
    if scale not in configs:
        raise ValueError(f"Unknown scale: {scale}. Choose from {list(configs.keys())}")
    return configs[scale]()
