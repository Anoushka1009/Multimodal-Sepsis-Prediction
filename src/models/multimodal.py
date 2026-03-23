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
from src.models.text_models import AttentionTextAggregator, TextWindowEncoder


class MultimodalClassifier(nn.Module):
    def __init__(
        self,
        structured_input_dim: int,
        text_embedding_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        structured_encoder_type: str = 'gru',
        fusion_strategy: str = 'gated_fusion',
    ):
        super().__init__()
        if structured_encoder_type == 'transformer':
            self.structured_encoder = TransformerStructuredEncoder(structured_input_dim, hidden_dim, dropout=dropout)
        else:
            self.structured_encoder = GRUStructuredEncoder(structured_input_dim, hidden_dim, dropout=dropout)

        self.text_encoder = TextWindowEncoder(text_embedding_dim, hidden_dim, dropout=dropout)
        self.text_attention = AttentionTextAggregator(text_embedding_dim, hidden_dim, dropout=dropout)
        self.fusion_strategy = fusion_strategy

        if fusion_strategy == 'early_fusion':
            self.fusion_head = EarlyFusionHead(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        elif fusion_strategy == 'late_fusion':
            self.fusion_head = LateFusionHead(hidden_dim, hidden_dim)
        elif fusion_strategy == 'cross_modal_attention':
            self.fusion_head = CrossModalAttentionFusion(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        else:
            self.fusion_head = GatedFusionHead(hidden_dim, hidden_dim, hidden_dim)

    def forward(self, structured_sequence: torch.Tensor, text_window_embeddings: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        structured_repr = self.structured_encoder(structured_sequence)
        text_repr, text_weights = self.text_attention(text_window_embeddings)
        aux = {'text_attention_weights': text_weights}

        if self.fusion_strategy == 'cross_modal_attention':
            logits, fusion_weights = self.fusion_head(structured_repr, text_repr)
            aux['fusion_attention_weights'] = fusion_weights
            return logits, aux

        logits = self.fusion_head(structured_repr, text_repr)
        return logits, aux
