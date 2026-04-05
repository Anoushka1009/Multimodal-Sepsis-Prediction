from __future__ import annotations

import math

import torch
from torch import nn


def _masked_mean(sequence: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    if padding_mask is None:
        return sequence.mean(dim=1)

    valid_mask = (~padding_mask).unsqueeze(-1).to(sequence.dtype)
    summed = (sequence * valid_mask).sum(dim=1)
    counts = valid_mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class GRUStructuredEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        outputs, hidden = self.gru(x)
        pooled = hidden[-1]

        if padding_mask is not None:
            valid_lengths = (~padding_mask).sum(dim=1).clamp(min=1)
            batch_indices = torch.arange(outputs.size(0), device=outputs.device)
            pooled = outputs[batch_indices, valid_lengths - 1]

        if return_sequence:
            return outputs, pooled
        return pooled


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerStructuredEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h = self.positional_encoding(h)
        h = self.input_dropout(h)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        pooled = _masked_mean(h, padding_mask)

        if return_sequence:
            return h, pooled
        return pooled
