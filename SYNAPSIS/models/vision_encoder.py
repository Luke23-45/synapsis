"""
SOTA Vision + Language Encoder

Replaces goal images with language instruction conditioning.
This is the practical SOTA approach used by Pi0, OpenVLA, and Octo.

Key Changes from v1:
- Removed goal_image dependency
- Added language instruction encoding via CLIP/T5
- Language embeddings fused via cross-attention

Why Language > Goal Images:
1. Goal images require capturing target state (impractical)
2. Humans naturally communicate via language
3. Language generalizes to novel objects via semantic understanding
4. SOTA models achieve 50+ Hz control with language conditioning
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Vision backbone
try:
    from transformers import SiglipModel, SiglipImageProcessor
    HAS_SIGLIP = True
except ImportError:
    HAS_SIGLIP = False

# Language encoder
try:
    from transformers import CLIPTextModel, CLIPTokenizer
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

try:
    from transformers import T5EncoderModel, T5Tokenizer
    HAS_T5 = True
except ImportError:
    HAS_T5 = False


class LanguageEncoder(nn.Module):
    """
    SOTA Language Encoder for instruction conditioning.
    
    Uses CLIP text encoder (same as OpenVLA) for efficiency,
    with optional T5 for richer semantic understanding.
    
    Args:
        model_name: HuggingFace model name
        d_model: Output dimension
        freeze: Whether to freeze language model
        max_length: Maximum instruction length
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        d_model: int = 768,
        freeze: bool = True,
        max_length: int = 77,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.freeze = freeze
        
        if HAS_CLIP and "clip" in model_name.lower():
            self.encoder = CLIPTextModel.from_pretrained(model_name)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.hidden_size = self.encoder.config.hidden_size  # 512 for CLIP
            self.encoder_type = "clip"
        elif HAS_T5 and "t5" in model_name.lower():
            self.encoder = T5EncoderModel.from_pretrained(model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.hidden_size = self.encoder.config.d_model  # 768 for t5-base
            self.encoder_type = "t5"
        else:
            # Fallback: simple embedding layer
            self.encoder = None
            self.tokenizer = None
            self.hidden_size = d_model
            self.encoder_type = "simple"
            self.vocab_size = 50000
            self.embedding = nn.Embedding(self.vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_length, d_model)
        
        # Freeze if requested
        if freeze and self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Output projection
        if self.hidden_size != d_model:
            self.proj = nn.Linear(self.hidden_size, d_model)
        else:
            self.proj = nn.Identity()
        
        # Global projection for conditioning
        self.global_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
    
    def forward(
        self,
        text: Union[str, List[str], torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode language instruction.
        
        Args:
            text: String, list of strings, or pre-tokenized input_ids (B, L)
            device: Target device
        
        Returns:
            tokens: (B, L, d_model) - Token embeddings for cross-attention
            global_emb: (B, d_model) - Global instruction embedding
        """
        # Handle string input
        if isinstance(text, str):
            text = [text]
        
        if isinstance(text, list):
            # Tokenize
            if self.tokenizer is not None:
                inputs = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                if device is not None:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
            else:
                # Simple fallback: hash words to indices
                B = len(text)
                input_ids = torch.zeros(B, self.max_length, dtype=torch.long)
                attention_mask = torch.zeros(B, self.max_length, dtype=torch.long)
                for i, t in enumerate(text):
                    words = t.lower().split()[:self.max_length]
                    for j, word in enumerate(words):
                        input_ids[i, j] = hash(word) % self.vocab_size
                        attention_mask[i, j] = 1
                if device is not None:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
        else:
            # Pre-tokenized tensor
            input_ids = text
            attention_mask = (input_ids != 0).long()
        
        # Encode
        if self.encoder is not None:
            if self.encoder_type == "clip":
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden = outputs.last_hidden_state  # (B, L, hidden)
            else:  # T5
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden = outputs.last_hidden_state
        else:
            # Simple embedding
            B, L = input_ids.shape
            pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
            hidden = self.embedding(input_ids) + self.pos_emb(pos)
        
        # Project to d_model
        tokens = self.proj(hidden)  # (B, L, d_model)
        
        # Global embedding: masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        global_emb = self.global_proj(pooled)  # (B, d_model)
        
        return tokens, global_emb


class SigLIPVisionEncoder(nn.Module):
    """
    SOTA Vision Encoder using SigLIP backbone.
    Unchanged from previous version.
    """
    
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        d_model: int = 768,
        unfreeze_last_n: int = 3,
        use_gradient_checkpointing: bool = True,
        num_views: int = 2,  # primary + wrist (no goal)
    ):
        super().__init__()
        
        if not HAS_SIGLIP:
            raise ImportError("transformers package required for SigLIP")
        
        self.d_model = d_model
        self.num_views = num_views
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Load pretrained SigLIP
        self.siglip = SiglipModel.from_pretrained(model_name)
        self.vision_model = self.siglip.vision_model
        self.config = self.vision_model.config
        
        self.hidden_size = self.config.hidden_size
        self.patch_size = self.config.patch_size
        self.image_size = self.config.image_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        # Freeze backbone except top-N layers
        self._freeze_backbone(unfreeze_last_n)
        
        # Gradient checkpointing
        if use_gradient_checkpointing:
            self.vision_model.encoder.gradient_checkpointing = True
        
        # Token-type embeddings: primary (0), wrist (1)
        self.token_type_emb = nn.Embedding(num_views, self.hidden_size)
        
        # Spatial position refinement
        self.spatial_pos_refine = nn.Parameter(
            torch.zeros(1, self.num_patches, self.hidden_size)
        )
        
        # Output projection
        if self.hidden_size != d_model:
            self.proj = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, d_model),
            )
        else:
            self.proj = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _freeze_backbone(self, unfreeze_last_n: int):
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        for param in self.vision_model.post_layernorm.parameters():
            param.requires_grad = True
        
        if unfreeze_last_n > 0:
            layers = self.vision_model.encoder.layers
            for layer in layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def _init_weights(self):
        nn.init.normal_(self.token_type_emb.weight, std=0.02)
        nn.init.zeros_(self.spatial_pos_refine)
    
    def _encode_single_view(
        self,
        pixel_values: torch.Tensor,
        view_type_id: int,
    ) -> torch.Tensor:
        B = pixel_values.shape[0]
        
        if self.use_gradient_checkpointing and self.training:
            outputs = checkpoint(
                self.vision_model,
                pixel_values,
                use_reentrant=False,
            )
        else:
            outputs = self.vision_model(pixel_values)
        
        hidden_states = outputs.last_hidden_state
        
        type_emb = self.token_type_emb(
            torch.full((B,), view_type_id, device=hidden_states.device, dtype=torch.long)
        ).unsqueeze(1)
        
        features = hidden_states + type_emb + self.spatial_pos_refine
        
        return features
    
    def forward(
        self,
        primary_image: torch.Tensor,
        wrist_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images (no goal image required).
        
        Args:
            primary_image: (B, C, H, W) - Primary camera
            wrist_image: (B, C, H, W) - Wrist camera (optional)
        
        Returns:
            context_tokens: (B, L, d_model)
            global_context: (B, d_model)
        """
        self._validate_input(primary_image, "primary_image")
        
        primary_features = self._encode_single_view(primary_image, view_type_id=0)
        all_features = [primary_features]
        
        if wrist_image is not None:
            self._validate_input(wrist_image, "wrist_image")
            wrist_features = self._encode_single_view(wrist_image, view_type_id=1)
            all_features.append(wrist_features)
        
        features = torch.cat(all_features, dim=1)
        context_tokens = self.proj(features)
        global_context = context_tokens.mean(dim=1)
        
        return context_tokens, global_context
    
    def _validate_input(self, tensor: torch.Tensor, name: str):
        if tensor.dim() != 4:
            raise ValueError(f"{name} must be 4D (B, C, H, W), got {tensor.shape}")
        if tensor.shape[1] != 3:
            raise ValueError(f"{name} must have 3 channels, got {tensor.shape[1]}")
        if tensor.shape[2] != self.image_size or tensor.shape[3] != self.image_size:
            raise ValueError(
                f"{name} must be {self.image_size}x{self.image_size}, "
                f"got {tensor.shape[2]}x{tensor.shape[3]}"
            )


class ProprioceptionEncoder(nn.Module):
    """Encodes proprioception into d_model space."""
    
    def __init__(
        self,
        proprio_dim: int = 22,
        d_model: int = 768,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.d_model = d_model
        
        self.encoder = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )
    
    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        if proprio.dim() != 2:
            raise ValueError(f"proprio must be 2D (B, D), got {proprio.shape}")
        return self.encoder(proprio)


class VisionLanguageEncoder(nn.Module):
    """
    Complete Vision + Language + Proprio Context Encoder.
    
    SOTA architecture following Pi0/OpenVLA:
    - SigLIP vision backbone (multi-view: primary + wrist)
    - CLIP/T5 language encoder (replaces goal images)
    - Proprioception encoder
    - Cross-modal fusion
    
    This is the practical, deployable SOTA approach.
    """
    
    def __init__(
        self,
        vision_backbone: str = "google/siglip-base-patch16-224",
        language_backbone: str = "openai/clip-vit-base-patch32",
        d_model: int = 768,
        proprio_dim: int = 22,
        unfreeze_vision_last_n: int = 3,
        freeze_language: bool = True,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.proprio_dim = proprio_dim
        
        # Vision encoder
        self.vision_encoder = SigLIPVisionEncoder(
            model_name=vision_backbone,
            d_model=d_model,
            unfreeze_last_n=unfreeze_vision_last_n,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        
        # Language encoder (replaces goal images)
        self.language_encoder = LanguageEncoder(
            model_name=language_backbone,
            d_model=d_model,
            freeze=freeze_language,
        )
        
        # Proprioception encoder
        self.proprio_encoder = ProprioceptionEncoder(
            proprio_dim=proprio_dim,
            d_model=d_model,
        )
        
        # Token type indicators
        self.proprio_token_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.language_token_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Context fusion
        self.context_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        
        # Global fusion (vision + language + proprio)
        self.global_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Uncond embedding for CFG
        self.uncond_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    
    def forward(
        self,
        primary_image: torch.Tensor,
        proprio: torch.Tensor,
        language_instruction: Union[str, List[str], torch.Tensor],
        wrist_image: Optional[torch.Tensor] = None,
        return_uncond: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode vision + language + proprioception.
        
        Args:
            primary_image: (B, 3, 224, 224) - Primary camera
            proprio: (B, proprio_dim) - Current proprioception
            language_instruction: Text instruction(s) or pre-tokenized tensor
            wrist_image: (B, 3, 224, 224) - Optional wrist camera
            return_uncond: If True, return uncond embeddings for CFG
        
        Returns:
            context_tokens: (B, L, d_model) - Cross-attention context
            global_context: (B, d_model) - Global conditioning
        """
        B = primary_image.shape[0]
        device = primary_image.device
        
        # Encode vision
        vision_tokens, vision_global = self.vision_encoder(
            primary_image=primary_image,
            wrist_image=wrist_image,
        )
        
        # Encode language (replaces goal images!)
        language_tokens, language_global = self.language_encoder(
            language_instruction,
            device=device,
        )
        language_tokens = language_tokens + self.language_token_type
        
        # Encode proprioception
        proprio_emb = self.proprio_encoder(proprio)
        proprio_token = proprio_emb.unsqueeze(1) + self.proprio_token_type
        
        # Concatenate all tokens for cross-attention
        # Order: vision tokens + language tokens + proprio token
        context_tokens = torch.cat([vision_tokens, language_tokens, proprio_token], dim=1)
        context_tokens = self.context_proj(context_tokens)
        
        # Global context: fuse vision + language + proprio
        global_context = self.global_fusion(
            torch.cat([vision_global, language_global, proprio_emb], dim=-1)
        )
        
        if return_uncond:
            uncond = self.uncond_embedding.expand(B, -1, -1)
            return context_tokens, global_context, uncond
        
        return context_tokens, global_context


class FallbackVisionLanguageEncoder(nn.Module):
    """
    Fallback encoder when transformers package is not available.
    Uses simple ConvNet + embeddings for development/testing.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        d_model: int = 768,
        proprio_dim: int = 22,
        vocab_size: int = 50000,
        max_text_len: int = 77,
    ):
        super().__init__()
        self.d_model = d_model
        self.proprio_dim = proprio_dim
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        
        # Simple vision encoder
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, d_model // 4, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv2d(d_model // 4, d_model // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((image_size // patch_size, image_size // patch_size)),
        )
        
        # Position embeddings
        self.vision_pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.token_type_emb = nn.Embedding(2, d_model)  # primary, wrist
        
        # Language encoder (simple)
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.text_pos_emb = nn.Embedding(max_text_len, d_model)
        
        # Proprio encoder
        self.proprio_encoder = ProprioceptionEncoder(proprio_dim, d_model)
        
        # Global projection
        self.global_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Token types
        self.proprio_token_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.language_token_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Uncond
        self.uncond_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    
    def _encode_view(self, img: torch.Tensor, view_type: int) -> torch.Tensor:
        B = img.shape[0]
        patches = self.patch_embed(img).flatten(2).transpose(1, 2)
        type_emb = self.token_type_emb(
            torch.full((B,), view_type, device=img.device, dtype=torch.long)
        ).unsqueeze(1)
        return patches + self.vision_pos_emb + type_emb
    
    def _encode_text(self, text: Union[str, List[str], torch.Tensor], device) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]
        
        if isinstance(text, list):
            B = len(text)
            input_ids = torch.zeros(B, self.max_text_len, dtype=torch.long, device=device)
            for i, t in enumerate(text):
                words = t.lower().split()[:self.max_text_len]
                for j, word in enumerate(words):
                    input_ids[i, j] = hash(word) % self.vocab_size
        else:
            input_ids = text
        
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        tokens = self.text_embedding(input_ids) + self.text_pos_emb(pos)
        return tokens + self.language_token_type
    
    def forward(
        self,
        primary_image: torch.Tensor,
        proprio: torch.Tensor,
        language_instruction: Union[str, List[str], torch.Tensor],
        wrist_image: Optional[torch.Tensor] = None,
        return_uncond: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = primary_image.shape[0]
        device = primary_image.device
        
        # Vision
        features = [self._encode_view(primary_image, 0)]
        if wrist_image is not None:
            features.append(self._encode_view(wrist_image, 1))
        vision_tokens = torch.cat(features, dim=1)
        
        # Language
        language_tokens = self._encode_text(language_instruction, device)
        
        # Proprio
        proprio_emb = self.proprio_encoder(proprio)
        proprio_token = proprio_emb.unsqueeze(1) + self.proprio_token_type
        
        # Combine
        context_tokens = torch.cat([vision_tokens, language_tokens, proprio_token], dim=1)
        
        # Global
        global_context = vision_tokens.mean(dim=1) + language_tokens.mean(dim=1) + proprio_emb
        global_context = self.global_proj(global_context)
        
        if return_uncond:
            uncond = self.uncond_embedding.expand(B, -1, -1)
            return context_tokens, global_context, uncond
        
        return context_tokens, global_context


def create_vision_encoder(
    vision_backbone: str = "google/siglip-base-patch16-224",
    language_backbone: str = "openai/clip-vit-base-patch32",
    d_model: int = 768,
    proprio_dim: int = 22,
    use_pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create Vision+Language encoder.
    
    Automatically falls back to simple encoder if dependencies unavailable.
    """
    if use_pretrained and HAS_SIGLIP:
        return VisionLanguageEncoder(
            vision_backbone=vision_backbone,
            language_backbone=language_backbone,
            d_model=d_model,
            proprio_dim=proprio_dim,
            **kwargs,
        )
    else:
        if use_pretrained:
            import warnings
            warnings.warn(
                "transformers package not available, using fallback encoder. "
                "Install with: pip install transformers"
            )
        return FallbackVisionLanguageEncoder(
            d_model=d_model,
            proprio_dim=proprio_dim,
        )
