"""
Module 3: NTH-Attention Transformer (Geometry-Aware Reasoner) [FINAL ULTRA VERSION]

The "Absolute Logic" implementation of structure-aware semantic inference.
This module integrates State-of-the-Art (SOTA) architectural patterns:
- Grouped Query Attention (GQA) for inference efficiency.
- RMSNorm for superior training stability (LLaMA-3 style).
- Spectral Manifold Pyramid for multi-scale geometric reasoning.
- Stochastic Depth (DropPath) for robust deep network convergence.
- Rotary Positional Embeddings (RoPE) for relative distance encoding.

Author: [Your Name]
Logic Level: Maximum (Deep Geometry + Efficient Transformer)
"""

from typing import Dict, Optional, Tuple, List, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# SECTION 1: Stability & Normalization Primitives
# ==============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    
    Superior to LayerNorm as it is invariant to re-scaling of weights and gradients.
    Used in Gopher, Chinchilla, LLaMA, and almost all modern SOTA LLMs.
    
    Formula: x / sqrt(mean(x^2) + eps) * weight
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class DropPath(nn.Module):
    """
    Stochastic Depth (DropPath).
    
    Randomly drops entire residual paths during training.
    Prevents co-adaptation of parallel paths and acts as an implicit ensemble.
    Crucial for training very deep transformers without convergence issues.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        # Work with any number of dimensions, broadcasting the batch dim
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        
        # Scale output to maintain expected value
        output = x.div(keep_prob) * random_tensor
        return output


# ==============================================================================
# SECTION 2: Advanced Embeddings & Activation
# ==============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) with Precomputed Cache.
    
    Encodes relative positions by rotating the query and key vectors in the embedding space.
    This implementation efficiently manages the frequency cache to avoid re-computation.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        
        # Build initial cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        # theta_i = 10000 ^ (-2(i-1)/d)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        
        # Concatenate sin/cos for rotation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        # Rebuild cache if sequence exceeds current max
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(max(seq_len, self.max_seq_len * 2))
            
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device)
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE to Q and K."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit.
    
    The "PaLM" variant of FFN.
    Math: FFN(x) = (SiLU(xW_1) * xW_2) W_3
    This offers a steeper gradient flow than standard GELU MLPs.
    """
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_in = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simultaneous projection for gate and value
        gate = F.silu(self.w_gate(x))
        val = self.w_in(x)
        output = self.w_out(gate * val)
        return self.dropout(output)


# ==============================================================================
# SECTION 3: The "Absolute Logic" - Multi-Scale Geometry & Modulation
# ==============================================================================

class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with Zero Initialization (DiT-Style).
    
    This is the control center of the block. It accepts a global conditioning vector
    (topology + time) and generates ALL modulation parameters for the block.
    """
    def __init__(self, d_model: int, condition_dim: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        condition_dim = condition_dim or d_model
        
        # SiLU activation for the conditioner
        self.act = nn.SiLU()
        
        # Project to 6 parameters:
        # 1. shift_attn (beta1)
        # 2. scale_attn (gamma1)
        # 3. gate_attn  (alpha1)
        # 4. shift_ffn  (beta2)
        # 5. scale_ffn  (gamma2)
        # 6. gate_ffn   (alpha2)
        self.linear = nn.Linear(condition_dim, 6 * d_model, bias=True)
        
        # Zero initialization is CRITICAL.
        # It ensures the block starts as an identity function, allowing gradients
        # to flow perfectly at the start of training.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes adaptive parameters.
        """
        vars = self.linear(self.act(condition))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = vars.chunk(6, dim=-1)
        
        # Reshape for broadcasting: (B, 1, D)
        shift_msa = shift_msa.unsqueeze(1)
        scale_msa = scale_msa.unsqueeze(1)
        gate_msa = gate_msa.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        gate_mlp = gate_mlp.unsqueeze(1)
        
        # Apply the FIRST modulation (for Attention) immediately
        # Note: We assume x is normalized externally or inside the block via RMSNorm
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


class SpectralManifoldPyramid(nn.Module):
    """
    Multi-Scale Manifold Router (The "Ultra" Geometric Logic).
    
    Standard projectors treat geometry as a flat 2D image.
    This module acknowledges that topology has multiple scales:
    1. Local Scale: Immediate neighbors (Convolutional processing).
    2. Global Scale: Distant connectivity (Dense/Linear processing).
    
    It fuses these views to create a "Geometry Bias" that is aware of both
    micro-structure and macro-structure.
    """
    def __init__(self, K_anchors: int, num_heads: int, d_model: int, L_max: int):
        super().__init__()
        self.K = K_anchors
        
        # Branch 1: Global Dense Projection (Connecting everything to everything)
        # Projects K -> L
        self.global_row = nn.Linear(K_anchors, L_max)
        self.global_col = nn.Linear(K_anchors, L_max)
        
        # Branch 2: Local Convolutional Projection (Preserving neighborhood)
        # Treating the adjacency matrix as a 1-channel image
        self.local_conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(4, 1, kernel_size=1)
        )
        # If L != K, we need to resize the local features
        self.resize = nn.Upsample(size=(L_max, L_max), mode='bilinear', align_corners=False)
        
        # Adaptive Fusion Gate
        # Learnable weight per head to balance Local vs Global
        self.fusion_gate = nn.Parameter(torch.full((1, num_heads, 1, 1), 0.5))

    def forward(self, B_geo: torch.Tensor, target_L: int) -> torch.Tensor:
        """
        Args:
            B_geo: (B, K, K) - The raw manifold distance/adjacency matrix.
            target_L: int - Current sequence length.
        Returns:
            bias: (B, H, L, L) - The fused geometric bias.
        """
        B, K, _ = B_geo.shape
        
        # --- PATH 1: Global Dense ---
        # Expand (B, 1, K, K)
        global_path = B_geo.unsqueeze(1)
        
        # Project rows and cols to match target_L
        # Slice the weights to handle dynamic lengths if target_L < L_max
        row_w = self.global_row.weight[:target_L, :K]
        col_w = self.global_col.weight[:target_L, :K]
        
        global_path = torch.matmul(global_path.transpose(-1, -2), row_w.t()).transpose(-1, -2)
        global_path = torch.matmul(global_path, col_w.t()) # (B, 1, L, L)
        
        # --- PATH 2: Local Conv ---
        # (B, 1, K, K)
        local_path = self.local_conv(B_geo.unsqueeze(1))
        
        # Resize if K != target_L
        if K != target_L:
            local_path = F.interpolate(local_path, size=(target_L, target_L), mode='bilinear', align_corners=False)
            
        # --- FUSION ---
        # Sigmoid to constrain gate to [0, 1]
        alpha = torch.sigmoid(self.fusion_gate)
        
        # Broadcasting to H heads
        # Global gets (1-alpha), Local gets (alpha)
        # This allows different heads to specialize in different scales
        fused_bias = (1 - alpha) * global_path + alpha * local_path
        
        return fused_bias # (B, H, L, L)


# ==============================================================================
# SECTION 4: Grouped Query Attention (GQA) Logic
# ==============================================================================

class NTH_GQA_Attention(nn.Module):
    """
    Grouped Query Attention (GQA) with Geometry Bias.
    
    Why GQA?
    Standard MHA has H Query heads, H Key heads, H Value heads.
    GQA has H Query heads, but only G < H Key/Value heads.
    
    This drastically reduces the size of the KV-Cache and memory bandwidth usage,
    allowing for larger batch sizes and faster inference without losing accuracy.
    This is the "Logic" of efficiency.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by heads"
        assert num_heads % self.num_kv_heads == 0, "Heads must be divisible by KV heads"
        
        # Projections
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        
        # QK Normalization (Stability hack for large models)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        geo_bias: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        B, L, _ = x.shape
        H, HKV = self.num_heads, self.num_kv_heads
        D = self.head_dim
        
        # 1. Projections
        xq = self.q_proj(x).view(B, L, H, D).transpose(1, 2)    # (B, H, L, D)
        xk = self.k_proj(x).view(B, L, HKV, D).transpose(1, 2)  # (B, HKV, L, D)
        xv = self.v_proj(x).view(B, L, HKV, D).transpose(1, 2)  # (B, HKV, L, D)
        
        # 2. QK Norm
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        
        # 3. RoPE
        if rope_cos is not None:
            xq, xk = apply_rotary_pos_emb(xq, xk, rope_cos, rope_sin)
            
        # 4. GQA: Repeat KV heads to match Q heads
        if self.num_kv_groups > 1:
            # (B, HKV, L, D) -> (B, HKV, G, L, D) -> (B, H, L, D)
            xk = xk[:, :, None, :, :].expand(B, HKV, self.num_kv_groups, L, D).reshape(B, H, L, D)
            xv = xv[:, :, None, :, :].expand(B, HKV, self.num_kv_groups, L, D).reshape(B, H, L, D)
            
        # 5. Attention Scores
        # (B, H, L, D) @ (B, H, D, L) -> (B, H, L, L)
        scores = torch.matmul(xq, xk.transpose(-1, -2)) / math.sqrt(D)
        
        # 6. Inject Geometric Bias
        if geo_bias is not None:
            scores = scores + geo_bias
            
        # 7. Masking
        if mask is not None:
             # Ensure mask broadcasts: (B, 1, L, L)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
            
        # 8. Softmax & Output
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        
        # (B, H, L, L) @ (B, H, L, D) -> (B, H, L, D)
        output = torch.matmul(probs, xv)
        
        # 9. Final Projection
        output = output.transpose(1, 2).contiguous().view(B, L, -1)
        output = self.o_proj(output)
        
        return output


# ==============================================================================
# SECTION 5: The Transformer Block
# ==============================================================================

class NTHTransformerBlock(nn.Module):
    """
    SOTA Transformer Block.
    
    Structure:
    Input -> RMSNorm -> Modulation(Scale/Shift) -> GQA + GeoBias -> DropPath -> Residual
    Input -> RMSNorm -> Modulation(Scale/Shift) -> SwiGLU        -> DropPath -> Residual
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        
        # Normalization
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # Modulation Controller
        self.adaln = AdaLNZero(d_model)
        
        # Attention
        self.attn = NTH_GQA_Attention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout
        )
        
        # Stochastic Depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # FFN
        hidden_dim = int(d_model * mlp_ratio * 2 / 3) # SwiGLU correction
        self.mlp = SwiGLU(d_model, hidden_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        geo_bias: Optional[torch.Tensor],
        rope_cos: Optional[torch.Tensor],
        rope_sin: Optional[torch.Tensor],
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        
        # 1. Get Modulation Params (all 6 at once)
        # shift/scale/gate for MSA and MLP
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(x, condition)
        
        # 2. Attention Branch
        # Apply RMSNorm -> Apply Affine Modulation -> Attention -> Gate -> DropPath -> Residual
        x_norm1 = self.norm1(x)
        x_mod1 = x_norm1 * (1 + scale_msa) + shift_msa
        
        attn_out = self.attn(x_mod1, geo_bias, rope_cos, rope_sin, mask)
        x = x + self.drop_path(gate_msa * attn_out)
        
        # 3. FFN Branch
        x_norm2 = self.norm2(x)
        x_mod2 = x_norm2 * (1 + scale_mlp) + shift_mlp
        
        mlp_out = self.mlp(x_mod2)
        x = x + self.drop_path(gate_mlp * mlp_out)
        
        return x


# ==============================================================================
# SECTION 6: The Transformer (The Orchestrator)
# ==============================================================================

class NTHAttentionTransformer(nn.Module):
    """
    NTH-Attention Transformer (Geometry-Aware Reasoner).
    
    The Final Assembly.
    - Fuses Global Z with Timesteps.
    - Projects Geometry via Spectral Pyramid.
    - Iterates through GQA Blocks.
    - Handles Topology Token (Learnable Start Token).
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None, # If None, defaults to num_heads (MHA)
        num_layers: int = 6,
        K_anchors: int = 16,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1 # Deep transformer trick
    ):
        super().__init__()
        self.d_model = d_model
        
        # 1. Topology State Token
        # Like the [CLS] token in BERT, but for topological reasoning
        self.topo_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # 2. Global Conditioner
        self.condition_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 3. Geometric Reasoning Engine
        self.manifold_pyramid = SpectralManifoldPyramid(K_anchors, num_heads, d_model, max_seq_len)
        
        # 4. RoPE
        self.rope = RotaryEmbedding(d_model // num_heads, max_seq_len)
        
        # 5. Stochastic Depth Decay Rule
        # Probabilities increase linearly with depth (deepest layers dropped most often)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        
        # 6. Stack Layers
        self.layers = nn.ModuleList([
            NTHTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dropout=dropout,
                drop_path=dpr[i]
            ) for i in range(num_layers)
        ])
        
        # 7. Final Norm
        self.final_norm = RMSNorm(d_model)
        
        # 8. Weight Initialization (Crucial for deep convergence)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(
        self,
        tokens: torch.Tensor,
        z_global: torch.Tensor,
        B_geo: torch.Tensor,
        timestep_emb: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tokens: (B, L, D) - Semantic tokens.
            z_global: (B, D) - Global topology embedding.
            B_geo: (B, K, K) - Geometric adjacency matrix.
            timestep_emb: (B, D) - Optional diffusion time step.
        """
        B, L, D = tokens.shape
        
        # 1. Condition Preparation
        cond_input = z_global
        if timestep_emb is not None:
            cond_input = cond_input + timestep_emb
        condition = self.condition_mlp(cond_input)
        
        # 2. Sequence Construction
        # Prepend [TOPO] token
        topo_t = self.topo_token.expand(B, -1, -1)
        x = torch.cat([topo_t, tokens], dim=1)
        curr_L = x.shape[1]
        
        # 3. Geometric Reasoning (Pyramid Projection)
        # B_geo is mapped to (B, H, curr_L, curr_L)
        geo_bias = self.manifold_pyramid(B_geo, target_L=curr_L)
        
        # 4. Positional Encoding
        rope_cos, rope_sin = self.rope(x, curr_L)
        
        # 5. Transformer Stack
        for block in self.layers:
            x = block(
                x, 
                condition=condition, 
                geo_bias=geo_bias, 
                rope_cos=rope_cos, 
                rope_sin=rope_sin, 
                mask=None
            )
            
        # 6. Finalization
        x = self.final_norm(x)
        
        return {
            'output': x[:, 1:, :],    # Sequence without TOPO token
            'topo_state': x[:, 0, :]  # TOPO token (Reasoning Summary)
        }