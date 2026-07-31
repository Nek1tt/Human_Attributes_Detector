"""Transformer classifier trained on MiniCPM visual embeddings."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .labels import ATTRIBUTE_SIZES_RU


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        divisor = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000) / d_model))
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.size(1) > self.pe.size(1):
            raise ValueError(f"Embedding sequence is longer than {self.pe.size(1)} tokens")
        return self.pe[:, : tensor.size(1), :]


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.probability == 0 or not self.training:
            return tensor
        keep = 1 - self.probability
        shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 1)
        mask = keep + torch.rand(shape, dtype=tensor.dtype, device=tensor.device)
        return tensor.div(keep) * mask.floor()


class PreNormEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(
        self, source: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        normalized = self.norm1(source)
        attended, _ = self.self_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        source = source + self.drop_path(self.dropout1(attended))
        normalized = self.norm2(source)
        projected = self.linear2(self.dropout(self.activation(self.linear1(normalized))))
        return source + self.drop_path(self.dropout2(projected))


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.attention(tensor).squeeze(-1)
        if mask is not None:
            if not mask.any(dim=1).all():
                raise ValueError("Every embedding sample must contain at least one valid token")
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (tensor * weights).sum(dim=1)


class VisionAttrTransformer(nn.Module):
    """Checkpoint-compatible classifier for tensors shaped ``[B, N, P, D]``."""

    def __init__(
        self,
        input_dim: int = 3584,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        attr_sizes: Mapping[str, int] | None = None,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        attr_sizes = attr_sizes or ATTRIBUTE_SIZES_RU
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.dropout = nn.Dropout(0.2)
        drop_rates = torch.linspace(0, drop_path_rate, num_layers).tolist()
        self.layers = nn.ModuleList(
            [
                PreNormEncoderLayer(
                    hidden_dim,
                    num_heads,
                    hidden_dim * 4,
                    dropout=0.1,
                    drop_path_rate=drop_rates[index],
                )
                for index in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention_pool = AttentionPooling(hidden_dim)
        self.heads = nn.ModuleDict(
            {
                attribute: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, size),
                )
                for attribute, size in attr_sizes.items()
            }
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def forward(
        self, embeddings: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if embeddings.ndim != 4:
            raise ValueError("embeddings must have shape [B, N, P, D]")
        batch, images, patches, dimension = embeddings.shape
        if dimension != self.input_proj.in_features:
            raise ValueError(
                f"Expected embedding dimension {self.input_proj.in_features}, got {dimension}"
            )
        flattened = embeddings.reshape(batch, images * patches, dimension)
        tensor = self.input_proj(flattened)
        tensor = tensor + self.pos_encoding(tensor)
        cls_token = self.cls_token.expand(batch, -1, -1)
        tensor = torch.cat([cls_token, tensor], dim=1)

        expanded_mask = None
        padding_mask = None
        if mask is not None:
            if mask.shape != (batch, images):
                raise ValueError(f"mask must have shape {(batch, images)}")
            expanded_mask = mask.unsqueeze(-1).expand(-1, -1, patches).reshape(batch, -1)
            cls_mask = torch.ones(batch, 1, dtype=torch.bool, device=tensor.device)
            padding_mask = ~torch.cat([cls_mask, expanded_mask], dim=1)

        tensor = self.dropout(tensor)
        for layer in self.layers:
            tensor = layer(tensor, src_key_padding_mask=padding_mask)
        tensor = self.norm(tensor)
        final = tensor[:, 0] + self.attention_pool(tensor[:, 1:], expanded_mask)
        return {attribute: head(final) for attribute, head in self.heads.items()}
