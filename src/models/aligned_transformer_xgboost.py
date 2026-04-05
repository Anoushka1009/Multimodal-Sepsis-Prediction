from __future__ import annotations

import torch
from torch import nn

from src.models.sequence_models import TransformerStructuredEncoder
from src.models.text_models import AttentionTextAggregator, BertTokenWindowEmbedder, TextWindowEncoder


def _masked_mean(sequence: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    if padding_mask is None:
        return sequence.mean(dim=1)

    valid_mask = (~padding_mask).unsqueeze(-1).to(sequence.dtype)
    summed = (sequence * valid_mask).sum(dim=1)
    counts = valid_mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def _safe_padding_mask(padding_mask: torch.Tensor | None) -> torch.Tensor | None:
    if padding_mask is None:
        return None
    safe_mask = padding_mask.clone()
    all_masked = safe_mask.all(dim=1)
    if all_masked.any():
        safe_mask[all_masked, 0] = False
    return safe_mask


class CrossModalAlignmentEncoder(nn.Module):
    def __init__(
        self,
        structured_dim: int,
        text_dim: int,
        hidden_dim: int,
        aligned_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
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
            nn.Linear(hidden_dim * 4, aligned_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        structured_sequence: torch.Tensor,
        text_sequence: torch.Tensor,
        structured_repr: torch.Tensor,
        text_repr: torch.Tensor,
        *,
        structured_padding_mask: torch.Tensor | None = None,
        text_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe_struct_padding_mask = _safe_padding_mask(structured_padding_mask)
        safe_text_padding_mask = _safe_padding_mask(text_padding_mask)

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
        return self.out(fused), attn_weights


class AlignedTransformerXGBoostEncoder(nn.Module):
    def __init__(
        self,
        *,
        structured_input_dim: int,
        text_embedding_dim: int = 768,
        structured_summary_dim: int = 0,
        hidden_dim: int = 128,
        aligned_dim: int = 256,
        dropout: float = 0.2,
        text_encoder_mode: str = "frozen_embedding",
        text_model_name: str | None = None,
        text_local_files_only: bool = False,
        text_finetune_unfrozen_layers: int = 2,
        text_gradient_checkpointing: bool = False,
        structured_num_heads: int = 4,
        structured_num_layers: int = 2,
        text_num_heads: int = 4,
        text_num_layers: int = 1,
        fusion_num_heads: int = 4,
    ):
        super().__init__()
        self.structured_encoder = TransformerStructuredEncoder(
            structured_input_dim,
            hidden_dim,
            num_heads=structured_num_heads,
            num_layers=structured_num_layers,
            dropout=dropout,
        )

        self.text_encoder_mode = str(text_encoder_mode).lower()
        self.text_window_embedder = None
        text_window_input_dim = int(text_embedding_dim)
        if self.text_encoder_mode == "bert_finetune":
            if not text_model_name:
                raise ValueError("text_model_name is required when text_encoder_mode='bert_finetune'.")
            self.text_window_embedder = BertTokenWindowEmbedder(
                model_name=text_model_name,
                local_files_only=bool(text_local_files_only),
                unfreeze_last_n=int(text_finetune_unfrozen_layers),
                gradient_checkpointing=bool(text_gradient_checkpointing),
            )
            text_window_input_dim = int(self.text_window_embedder.embedding_dim)

        self.text_encoder = TextWindowEncoder(
            text_window_input_dim,
            hidden_dim,
            num_heads=text_num_heads,
            num_layers=text_num_layers,
            dropout=dropout,
        )
        self.text_attention = AttentionTextAggregator(hidden_dim, dropout=dropout)

        self.summary_encoder = None
        self.structured_context = None
        if int(structured_summary_dim) > 0:
            self.summary_encoder = nn.Sequential(
                nn.Linear(structured_summary_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.structured_context = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.alignment_encoder = CrossModalAlignmentEncoder(
            structured_dim=hidden_dim,
            text_dim=hidden_dim,
            hidden_dim=hidden_dim,
            aligned_dim=aligned_dim,
            num_heads=fusion_num_heads,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(aligned_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        structured_sequence: torch.Tensor,
        *,
        text_window_embeddings: torch.Tensor | None = None,
        text_input_ids: torch.Tensor | None = None,
        text_token_attention_mask: torch.Tensor | None = None,
        structured_summary_features: torch.Tensor | None = None,
        structured_padding_mask: torch.Tensor | None = None,
        text_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        structured_sequence_repr, structured_repr = self.structured_encoder(
            structured_sequence,
            padding_mask=structured_padding_mask,
            return_sequence=True,
        )

        if self.text_window_embedder is not None:
            if text_input_ids is None or text_token_attention_mask is None:
                raise ValueError("Tokenized text inputs are required for text_encoder_mode='bert_finetune'.")
            text_window_representations = self.text_window_embedder(text_input_ids, text_token_attention_mask)
        else:
            if text_window_embeddings is None:
                raise ValueError("text_window_embeddings are required for frozen embedding mode.")
            text_window_representations = text_window_embeddings

        text_sequence_repr = self.text_encoder(text_window_representations, padding_mask=text_padding_mask)
        text_repr, text_weights = self.text_attention(text_sequence_repr, padding_mask=text_padding_mask)

        aux = {"text_attention_weights": text_weights}
        if self.summary_encoder is not None and structured_summary_features is not None:
            summary_repr = self.summary_encoder(structured_summary_features)
            structured_repr = self.structured_context(torch.cat([structured_repr, summary_repr], dim=-1))
            aux["structured_summary_repr"] = summary_repr

        aligned_repr, fusion_weights = self.alignment_encoder(
            structured_sequence_repr,
            text_sequence_repr,
            structured_repr,
            text_repr,
            structured_padding_mask=structured_padding_mask,
            text_padding_mask=text_padding_mask,
        )
        aux["aligned_repr"] = aligned_repr
        aux["fusion_attention_weights"] = fusion_weights

        logits = self.classifier(aligned_repr).squeeze(-1)
        return logits, aux
