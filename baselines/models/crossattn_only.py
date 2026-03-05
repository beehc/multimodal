"""B6: Cross-attention fusion baseline (no GMU).

Identical to the main multimodal model but replaces the Gated Multimodal
Unit with a simple concatenation, isolating the effect of the gating
mechanism.
"""

import torch
import torch.nn as nn

from src.model import CrossAttentionModule, ResNet1D, TabularEncoder, TextEncoder


class CrossAttnOnlyModel(nn.Module):
    """Cross-attention fusion without the Gated Multimodal Unit.

    Architecture:
        TextEncoder(256) + ResNet1D(256) → CrossAttentionModule → 512
        concat(cross_attn(512), tabular(64)) = 576
        MLP classifier: 576→256→128→3

    Args:
        num_classes: Number of output classes.
        dropout: Dropout probability for the classification head.
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.text_encoder = TextEncoder()       # → (batch, 256)
        self.ecg_encoder = ResNet1D()            # → (batch, 256)
        self.tabular_encoder = TabularEncoder()  # → (batch, 64)
        self.cross_attention = CrossAttentionModule(embed_dim=256, num_heads=8)

        fused_dim = 512 + 64  # cross-attn output + tabular

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ecg: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)``.
            attention_mask: Attention mask of shape ``(batch, seq_len)``.
            ecg: 12-lead ECG tensor of shape ``(batch, 12, seq_len)``.
            tabular: Tabular features of shape ``(batch, 9)``.

        Returns:
            Logits of shape ``(batch, 3)``.
        """
        text_features = self.text_encoder(input_ids, attention_mask)  # (batch, 256)
        ecg_features = self.ecg_encoder(ecg)                          # (batch, 256)
        tabular_features = self.tabular_encoder(tabular)              # (batch, 64)

        cross_attn_out = self.cross_attention(
            text_features, ecg_features,
        )  # (batch, 512)

        fused = torch.cat([cross_attn_out, tabular_features], dim=1)  # (batch, 576)
        return self.classifier(fused)
