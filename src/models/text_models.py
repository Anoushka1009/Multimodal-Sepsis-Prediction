from __future__ import annotations

import torch
from torch import nn


class TextWindowEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, window_embeddings: torch.Tensor) -> torch.Tensor:
        projected = self.proj(window_embeddings)
        return projected.mean(dim=1)


class AttentionTextAggregator(nn.Module):
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, window_embeddings: torch.Tensor):
        h = torch.tanh(self.proj(window_embeddings))
        h = self.dropout(h)
        scores = self.score(h).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)
        return pooled, weights
