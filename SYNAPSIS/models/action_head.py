"""
Flow Matching Action Head

SOTA action generation using Flow Matching (Optimal Transport paths).
Replaces DDIM diffusion with faster, more deterministic inference.

Key Components:
- SinusoidalPosEmb: Timestep embedding
- FlowMatchingActionHead: Velocity prediction network
"""

import math
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# EMBEDDED V1 TRANSFORMER BLOCK (Self-Contained for Stability)
# Required because V2 NTHTransformerBlock dropped cross-attention support
# which action_head.py strictly requires.
# ==============================================================================

from typing import Dict, Optional, Tuple, Union, List

class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization with Zero Initialization (V1)."""
    def __init__(self, d_model: int, condition_dim: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        condition_dim = condition_dim or d_model
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 3 * d_model),
        )
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mod = self.modulation(condition)
        scale, shift, gate = mod.chunk(3, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        gate = gate.unsqueeze(1)
        x_norm = self.norm(x)
        return x_norm * (1 + scale) + shift, gate


class GeometricBiasProjection(nn.Module):
    """Projects KxK geometric bias to attention-compatible dimensions (V1)."""
    def __init__(self, K: int, num_heads: int, L: Optional[int] = None):
        super().__init__()
        self.K = K
        self.num_heads = num_heads
        self.L = L or K
        self.head_weights = nn.Parameter(torch.randn(num_heads, 1, 1) * 0.01)
        if self.L != K:
            self.spatial_transform = nn.Upsample(size=(self.L, self.L), mode='bilinear', align_corners=False)
        else:
            self.spatial_transform = None
    
    def forward(self, B_geo: torch.Tensor) -> torch.Tensor:
        bias = B_geo.unsqueeze(1)
        if self.spatial_transform is not None:
            bias = self.spatial_transform(bias)
        return bias * self.head_weights


class NTHMultiHeadAttention(nn.Module):
    """Geometry-Biased Multi-Head Attention (V1)."""
    def __init__(self, d_model: int = 512, num_heads: int = 8, K: int = 16, L: Optional[int] = None, dropout: float = 0.1, lambda_init: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.K = K
        self.L = L or K
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.geo_proj = GeometricBiasProjection(K, num_heads, self.L)
        self.lambda_geo = nn.Parameter(torch.full((num_heads, 1, 1), lambda_init))
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, B_geo: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = x.shape
        H = self.num_heads
        q = self.q_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if B_geo is not None:
            geo_bias = self.geo_proj(B_geo)
            attn_scores = attn_scores + self.lambda_geo * geo_bias
        if attention_mask is not None:
            if attention_mask.dim() == 2: attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        attn_weights = self.attn_dropout(F.softmax(attn_scores, dim=-1))
        output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_dropout(self.out_proj(output)), None


class NTHTransformerBlock(nn.Module):
    """Single block of the NTH-Attention Transformer (V1)."""
    def __init__(self, d_model: int = 512, num_heads: int = 8, K: int = 16, dropout: float = 0.1, ffn_ratio: float = 4.0, use_cross_attention: bool = True, lambda_init: float = 0.5):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.adaln1 = AdaLNZero(d_model)
        self.self_attn = NTHMultiHeadAttention(d_model, num_heads, K, dropout=dropout, lambda_init=lambda_init)
        if use_cross_attention:
            self.adaln2 = AdaLNZero(d_model)
            self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.adaln3 = AdaLNZero(d_model)
        ffn_dim = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model), nn.Dropout(dropout))
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor, B_geo: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm, gate1 = self.adaln1(x, condition)
        attn_out, _ = self.self_attn(x_norm, B_geo=B_geo)
        x = x + gate1 * attn_out
        if self.use_cross_attention and context is not None:
            x_norm, gate2 = self.adaln2(x, condition)
            cross_out, _ = self.cross_attn(x_norm, context, context)
            x = x + gate2 * cross_out
        x_norm, gate3 = self.adaln3(x, condition)
        x = x + gate3 * self.ffn(x_norm)
        return x


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal position embeddings for timesteps.
    
    Used to encode diffusion/flow timestep as a continuous embedding.
    Same approach as in Transformers and Diffusion Models.
    """
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) - Timestep values in [0, 1]
        
        Returns:
            emb: (B, dim) - Sinusoidal embedding
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device) / half
        )
        
        # Scale t to match diffusion convention if needed
        args = t.unsqueeze(-1) * freqs.unsqueeze(0) * self.max_period
        
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        
        return emb


class FlowMatchingActionHead(nn.Module):
    """
    Flow Matching head for action generation.
    
    Replaces DDIM diffusion with optimal transport flow matching
    for faster and more deterministic sampling.
    
    Key Insight:
    - Flow Matching learns velocity field along straight paths
    - Single-step training, multi-step inference
    - Faster convergence than diffusion
    
    Reference: "Flow Matching for Generative Modeling" (ICML 2023)
               "π0: A Vision-Language-Action Flow Model" (Physical Intelligence, 2024)
    """
    
    def __init__(
        self,
        d_model: int = 512,
        action_dim: int = 8,
        action_chunk_size: int = 8,
        num_layers: int = 4,
        num_heads: int = 8,
        K: int = 16,
        dropout: float = 0.1,
        ffn_ratio: float = 4.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        
        # Action embedding
        self.action_proj = nn.Linear(action_dim, d_model)
        
        # Positional embedding for action sequence
        self.pos_embed = nn.Parameter(
            torch.randn(1, action_chunk_size, d_model) * 0.02
        )
        
        # Time embedding (sinusoidal + MLP)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        # Transformer decoder blocks (using NTH blocks for geometric awareness)
        # NOTE: Using embedded V1 block because V2 dropped cross-attention support
        self.blocks = nn.ModuleList([
            NTHTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                K=action_chunk_size,  # Actions are their own "anchors"
                dropout=dropout,
                ffn_ratio=ffn_ratio,
                use_cross_attention=True,
                lambda_init=0.0,  # Start without geometric bias in action head
            )
            for _ in range(num_layers)
        ])
        
        # Output projection (predict velocity, not noise)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )
        
        # SOTA: Initialize blocks and output projection properly
        self._init_sota_weights()
    
    def _init_sota_weights(self):
        """
        SOTA Initializations from DiT/Pi0:
        1. Initialize all block AdaLN gates as zero (identity at start)
        2. Initialize final output projection as zero (predict zero velocity at start)
        """
        # Linear projections for actions and time
        nn.init.xavier_uniform_(self.action_proj.weight)
        
        # Output projection: zero-init for stability
        if isinstance(self.out_proj[-1], nn.Linear):
            nn.init.zeros_(self.out_proj[-1].weight)
            nn.init.zeros_(self.out_proj[-1].bias)
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        condition: torch.Tensor,
        B_geo: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity at interpolation point x_t.
        
        Args:
            x_t: (B, K_act, D_act) - Interpolated actions at time t
            t: (B,) - Interpolation time [0, 1]
            context: (B, L, D) - Cross-attention context (vision tokens)
            condition: (B, D) - AdaLN condition (z_global + time)
            B_geo: (B, K, K) - Optional geometric bias
        
        Returns:
            v_pred: (B, K_act, D_act) - Predicted velocity
        """
        B, K, D = x_t.shape
        
        # Embed actions
        h = self.action_proj(x_t)  # (B, K, d_model)
        h = h + self.pos_embed[:, :K, :]
        
        # Embed time and combine with condition
        t_emb = self.time_embed(t)  # (B, d_model)
        combined_cond = condition + t_emb
        
        # Interpolate geometric bias if dimensions mismatch (e.g. anchors=16 vs actions=8)
        if B_geo is not None:
             k_geo = B_geo.shape[-1]
             if k_geo != K:
                 # Reshape (B, K_geo, K_geo) -> (B, 1, K_geo, K_geo) -> interpolate -> (B, K_act, K_act)
                 B_geo = F.interpolate(
                     B_geo.unsqueeze(1), 
                     size=(K, K), 
                     mode='bilinear', 
                     align_corners=False
                 ).squeeze(1)

        # Apply transformer blocks
        for block in self.blocks:
            h = block(h, condition=combined_cond, B_geo=B_geo, context=context)
        
        # Project to velocity
        v_pred = self.out_proj(h)  # (B, K, D_act)
        
        return v_pred
    
    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        condition: torch.Tensor,
        num_steps: int = 10,
        B_geo: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample actions via flow matching ODE (Euler integration).
        
        Args:
            context: (B, L, D) - Vision context
            condition: (B, D) - Topology condition (z_global)
            num_steps: Number of integration steps
            B_geo: (B, K, K) - Optional geometric bias
        
        Returns:
            actions: (B, K_act, D_act) - Sampled actions
        """
        B = context.shape[0]
        device = context.device
        
        # Start from noise
        x = torch.randn(B, self.action_chunk_size, self.action_dim, device=device)
        
        # Euler integration from t=0 to t=1
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=device)
            v = self.forward(x, t, context, condition, B_geo)
            x = x + v * dt
        
        return x


class ActionNormalizer(nn.Module):
    """
    Normalizes actions to [-1, 1] range for stable training.
    
    Critical for diffusion/flow policies to prevent NaN issues.
    Supports fitting from data or pre-computed statistics.
    """
    
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        
        # Register buffers for stats (moved with model to device)
        self.register_buffer('mean', torch.zeros(action_dim))
        self.register_buffer('std', torch.ones(action_dim))
        self.register_buffer('min_val', torch.zeros(action_dim))
        self.register_buffer('max_val', torch.ones(action_dim))
        self.register_buffer('fitted', torch.tensor(False))
    
    def fit(self, actions: torch.Tensor):
        """
        Fit normalizer from action data.
        
        Args:
            actions: (N, D) or (N, T, D) - Action samples
        """
        if actions.dim() == 3:
            actions = actions.reshape(-1, actions.shape[-1])
        
        self.mean.copy_(actions.mean(dim=0))
        self.std.copy_(actions.std(dim=0).clamp(min=1e-6))
        self.min_val.copy_(actions.min(dim=0).values)
        self.max_val.copy_(actions.max(dim=0).values)
        self.fitted.fill_(True)
    
    def fit_from_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        min_val: Optional[torch.Tensor] = None,
        max_val: Optional[torch.Tensor] = None,
    ):
        """Fit from pre-computed statistics."""
        self.mean.copy_(mean)
        self.std.copy_(std.clamp(min=1e-6))
        if min_val is not None:
            self.min_val.copy_(min_val)
        if max_val is not None:
            self.max_val.copy_(max_val)
        self.fitted.fill_(True)
    
    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Normalize actions to approximately [-1, 1].
        
        Uses z-score normalization with std-based clipping.
        """
        normalized = (actions - self.mean) / self.std
        # Clip to prevent extreme values
        normalized = normalized.clamp(-5, 5) / 5  # Map to [-1, 1]
        return normalized
    
    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Reverse normalization."""
        actions = actions * 5  # Reverse clipping scale
        return actions * self.std + self.mean
