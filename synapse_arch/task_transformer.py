from __future__ import annotations

import torch
from torch import nn


class TaskTransformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dropout: float, ffn_ratio: int, max_tokens: int) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ffn_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None, pos_indices: torch.Tensor | None = None) -> torch.Tensor:
        steps = tokens.shape[1]
        
        if pos_indices is not None:
            # pos_indices: (batch, seq_len)
            batch_size = tokens.shape[0]
            pos_expanded = self.pos_embed.expand(batch_size, -1, -1)
            # Gather positional embeddings based on indices
            # pos_indices shape: (batch_size, seq_len, 1) -> expanded to (batch_size, seq_len, d_model)
            pos = torch.gather(pos_expanded, 1, pos_indices.unsqueeze(-1).expand(-1, -1, self.pos_embed.shape[-1]))
            encoded = tokens + pos
        else:
            if steps > self.max_tokens:
                raise ValueError(f"Token sequence length {steps} exceeds max_tokens {self.max_tokens}")
            encoded = tokens + self.pos_embed[:, :steps, :]
            
        return self.encoder(encoded, src_key_padding_mask=key_padding_mask)
