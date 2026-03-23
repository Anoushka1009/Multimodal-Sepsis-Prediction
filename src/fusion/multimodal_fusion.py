from __future__ import annotations

import torch
from torch import nn


class EarlyFusionHead(nn.Module):
    def __init__(self, structured_dim: int, text_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(structured_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, structured_repr: torch.Tensor, text_repr: torch.Tensor):
        fused = torch.cat([structured_repr, text_repr], dim=-1)
        return self.net(fused).squeeze(-1)


class LateFusionHead(nn.Module):
    def __init__(self, structured_dim: int, text_dim: int):
        super().__init__()
        self.structured_head = nn.Linear(structured_dim, 1)
        self.text_head = nn.Linear(text_dim, 1)

    def forward(self, structured_repr: torch.Tensor, text_repr: torch.Tensor):
        return 0.5 * self.structured_head(structured_repr).squeeze(-1) + 0.5 * self.text_head(text_repr).squeeze(-1)


class GatedFusionHead(nn.Module):
    def __init__(self, structured_dim: int, text_dim: int, hidden_dim: int):
        super().__init__()
        self.structured_proj = nn.Linear(structured_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, structured_repr: torch.Tensor, text_repr: torch.Tensor):
        s = torch.tanh(self.structured_proj(structured_repr))
        t = torch.tanh(self.text_proj(text_repr))
        gate = torch.sigmoid(self.gate(torch.cat([s, t], dim=-1)))
        fused = gate * s + (1.0 - gate) * t
        return self.out(fused).squeeze(-1)


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, structured_dim: int, text_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.structured_proj = nn.Linear(structured_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, structured_repr: torch.Tensor, text_repr: torch.Tensor):
        query = self.structured_proj(structured_repr).unsqueeze(1)
        key_value = self.text_proj(text_repr).unsqueeze(1)
        attended, attn_weights = self.attention(query, key_value, key_value)
        logits = self.out(attended.squeeze(1)).squeeze(-1)
        return logits, attn_weights
