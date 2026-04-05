from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from src.fusion.multimodal_fusion import (
    CrossModalAttentionFusion,
    EarlyFusionHead,
    GatedFusionHead,
    LateFusionHead,
)
from src.models.sequence_models import GRUStructuredEncoder, TransformerStructuredEncoder
from src.models.text_models import AttentionTextAggregator, BertTokenWindowEmbedder, TextWindowEncoder


class MultimodalClassifier(nn.Module):
    def __init__(
        self,
        structured_input_dim: int,
        text_embedding_dim: int = 768,
        structured_summary_dim: int = 0,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        structured_encoder_type: str = 'gru',
        text_encoder_mode: str = 'frozen_embedding',
        text_model_name: str | None = None,
        text_local_files_only: bool = False,
        text_finetune_unfrozen_layers: int = 2,
        text_gradient_checkpointing: bool = False,
        fusion_strategy: str = 'gated_fusion',
        structured_num_heads: int = 4,
        structured_num_layers: int = 2,
        text_num_heads: int = 4,
        text_num_layers: int = 1,
        fusion_num_heads: int = 4,
    ):
        super().__init__()
        if structured_encoder_type == 'transformer':
            self.structured_encoder = TransformerStructuredEncoder(
                structured_input_dim,
                hidden_dim,
                num_heads=structured_num_heads,
                num_layers=structured_num_layers,
                dropout=dropout,
            )
        else:
            self.structured_encoder = GRUStructuredEncoder(structured_input_dim, hidden_dim, dropout=dropout)

        self.text_encoder_mode = str(text_encoder_mode).lower()
        self.text_window_embedder = None
        text_window_input_dim = int(text_embedding_dim)
        if self.text_encoder_mode == 'bert_finetune':
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
        self.fusion_strategy = fusion_strategy

        if fusion_strategy == 'early_fusion':
            self.fusion_head = EarlyFusionHead(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        elif fusion_strategy == 'late_fusion':
            self.fusion_head = LateFusionHead(hidden_dim, hidden_dim)
        elif fusion_strategy == 'cross_modal_attention':
            self.fusion_head = CrossModalAttentionFusion(
                hidden_dim,
                hidden_dim,
                hidden_dim,
                num_heads=fusion_num_heads,
                dropout=dropout,
            )
        else:
            self.fusion_head = GatedFusionHead(hidden_dim, hidden_dim, hidden_dim)

    def forward(
        self,
        structured_sequence: torch.Tensor,
        text_window_embeddings: torch.Tensor | None = None,
        text_input_ids: torch.Tensor | None = None,
        text_token_attention_mask: torch.Tensor | None = None,
        structured_summary_features: torch.Tensor | None = None,
        structured_padding_mask: torch.Tensor | None = None,
        text_padding_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, dict]:
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
        aux = {'text_attention_weights': text_weights}

        if self.summary_encoder is not None and structured_summary_features is not None:
            summary_repr = self.summary_encoder(structured_summary_features)
            structured_repr = self.structured_context(torch.cat([structured_repr, summary_repr], dim=-1))
            aux['structured_summary_repr'] = summary_repr

        if self.fusion_strategy == 'cross_modal_attention':
            logits, fusion_weights = self.fusion_head(
                structured_sequence_repr,
                text_sequence_repr,
                structured_repr,
                text_repr,
                structured_padding_mask=structured_padding_mask,
                text_padding_mask=text_padding_mask,
            )
            aux['fusion_attention_weights'] = fusion_weights
            return logits, aux

        logits = self.fusion_head(structured_repr, text_repr)
        return logits, aux
