from __future__ import annotations

import torch
from torch import nn


def _masked_mean(sequence: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    if padding_mask is None:
        return sequence.mean(dim=1)

    valid_mask = (~padding_mask).unsqueeze(-1).to(sequence.dtype)
    summed = (sequence * valid_mask).sum(dim=1)
    counts = valid_mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


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
        self.structured_to_text = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.text_to_struct = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.structured_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        structured_sequence: torch.Tensor,
        text_sequence: torch.Tensor,
        structured_repr: torch.Tensor,
        text_repr: torch.Tensor,
        structured_padding_mask: torch.Tensor | None = None,
        text_padding_mask: torch.Tensor | None = None,
    ):
        safe_struct_padding_mask = structured_padding_mask
        if safe_struct_padding_mask is not None:
            safe_struct_padding_mask = safe_struct_padding_mask.clone()
            all_masked = safe_struct_padding_mask.all(dim=1)
            if all_masked.any():
                safe_struct_padding_mask[all_masked, 0] = False

        safe_text_padding_mask = text_padding_mask
        if safe_text_padding_mask is not None:
            safe_text_padding_mask = safe_text_padding_mask.clone()
            all_masked = safe_text_padding_mask.all(dim=1)
            if all_masked.any():
                safe_text_padding_mask[all_masked, 0] = False

        structured_hidden = self.structured_proj(structured_sequence)
        text_hidden = self.text_proj(text_sequence)

        structured_attended, attn_weights = self.structured_to_text(
            query=structured_hidden,
            key=text_hidden,
            value=text_hidden,
            key_padding_mask=safe_text_padding_mask,
        )
        text_attended, _ = self.text_to_struct(
            query=text_hidden,
            key=structured_hidden,
            value=structured_hidden,
            key_padding_mask=safe_struct_padding_mask,
        )

        structured_context = self.structured_norm(structured_hidden + structured_attended)
        text_context = self.text_norm(text_hidden + text_attended)

        structured_summary = _masked_mean(structured_context, structured_padding_mask)
        text_summary = _masked_mean(text_context, text_padding_mask)
        fused = torch.cat([structured_summary, text_summary, structured_repr, text_repr], dim=-1)
        logits = self.out(fused).squeeze(-1)
        return logits, attn_weights
