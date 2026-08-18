from typing import Any

import torch
import torch.nn as nn


def _resolve_activation(name: str) -> nn.Module:
    normalized = name.upper()
    if normalized == "GELU":
        return nn.GELU()
    if normalized == "RELU":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


class AttentionAwareEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        activation: str = "GELU",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
    ) -> None:
        if d_model <= 0 or num_heads <= 0 or ffn_dim <= 0:
            raise ValueError("d_model, num_heads and ffn_dim must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        if norm_first:
            raise ValueError("TRANSFORMER-v1 uses post-norm encoder layers")
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout_attn_residual = nn.Dropout(dropout)
        self.dropout_ff_hidden = nn.Dropout(dropout)
        self.dropout_ff_residual = nn.Dropout(dropout)
        self.activation = _resolve_activation(activation)
        self.d_model = d_model
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, D]")
        attn_output, attention_weights = self.self_attn(
            x,
            x,
            x,
            need_weights=return_attention,
            average_attn_weights=False if return_attention else True,
            attn_mask=None,
            key_padding_mask=None,
            is_causal=False,
        )
        x = self.norm1(x + self.dropout_attn_residual(attn_output))
        ff_hidden = self.dropout_ff_hidden(self.activation(self.linear1(x)))
        x = self.norm2(x + self.dropout_ff_residual(self.linear2(ff_hidden)))
        if return_attention:
            if attention_weights is None:
                raise RuntimeError("Attention weights missing despite return_attention=True")
            return x, attention_weights
        return x, None


class AttentionAwareTransformerEncoder(nn.Module):
    def __init__(self, layers: list[AttentionAwareEncoderLayer]) -> None:
        if not layers:
            raise ValueError("Encoder requires at least one layer")
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attention_maps: list[torch.Tensor] = []
        hidden = x
        for layer in self.layers:
            hidden, attention = layer(hidden, return_attention=return_attention)
            if return_attention and attention is not None:
                attention_maps.append(attention)
        if return_attention and len(attention_maps) != len(self.layers):
            raise RuntimeError("Attention layer count mismatch")
        return hidden, attention_maps
