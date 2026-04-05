from __future__ import annotations

import torch
from torch import nn

from src.models.sequence_models import PositionalEncoding


class BertTokenWindowEmbedder(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        local_files_only: bool,
        unfreeze_last_n: int = 2,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.embedding_dim = int(self.backbone.config.hidden_size)
        self.cls_token_id = int(getattr(self.backbone.config, "cls_token_id", 101) or 101)
        self.is_finetuning_enabled = True

        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        encoder = getattr(self.backbone, "encoder", None)
        layers = list(getattr(encoder, "layer", [])) if encoder is not None else []
        if int(unfreeze_last_n) > 0 and layers:
            for layer in layers[-int(unfreeze_last_n) :]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        pooler = getattr(self.backbone, "pooler", None)
        if pooler is not None:
            for parameter in pooler.parameters():
                parameter.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 3 or attention_mask.ndim != 3:
            raise ValueError("BERT token window embedder expects [batch, windows, tokens] tensors.")

        batch_size, num_windows, max_tokens = input_ids.shape
        flat_input_ids = input_ids.reshape(batch_size * num_windows, max_tokens)
        flat_attention_mask = attention_mask.reshape(batch_size * num_windows, max_tokens)

        all_padded = flat_attention_mask.sum(dim=1) == 0
        if all_padded.any():
            flat_input_ids = flat_input_ids.clone()
            flat_attention_mask = flat_attention_mask.clone()
            flat_input_ids[all_padded, 0] = self.cls_token_id
            flat_attention_mask[all_padded, 0] = 1

        outputs = self.backbone(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        hidden = outputs.last_hidden_state
        mask = flat_attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled.reshape(batch_size, num_windows, self.embedding_dim)


class TextWindowEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=64)
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
        window_embeddings: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        effective_padding_mask = padding_mask
        if effective_padding_mask is not None:
            effective_padding_mask = effective_padding_mask.clone()
            all_masked = effective_padding_mask.all(dim=1)
            if all_masked.any():
                effective_padding_mask[all_masked, 0] = False

        h = self.input_proj(window_embeddings)
        h = self.positional_encoding(h)
        h = self.input_dropout(h)
        h = self.encoder(h, src_key_padding_mask=effective_padding_mask)
        return h


class AttentionTextAggregator(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        window_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ):
        h = torch.tanh(window_states)
        h = self.dropout(h)
        scores = self.score(h).squeeze(-1)

        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=1)
        if padding_mask is not None:
            valid = (~padding_mask).to(weights.dtype)
            weights = weights * valid
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        pooled = torch.sum(window_states * weights.unsqueeze(-1), dim=1)
        return pooled, weights
