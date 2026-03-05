"""B4: Early (concatenation) fusion baseline.

Encodes each modality independently, concatenates the representations, and
classifies with a shared MLP head.  No cross-attention or gating is used.
"""

import torch
import torch.nn as nn

from src.model import ResNet1D, TabularEncoder, TextEncoder


class EarlyFusionModel(nn.Module):
    """Simple early-fusion model via feature concatenation.

    Architecture:
        TextEncoder(256) + ResNet1D(256) + TabularEncoder(64)
        → concat(576) → MLP 576→256→128→3

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

        fused_dim = 256 + 256 + 64  # 576

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

        fused = torch.cat([text_features, ecg_features, tabular_features], dim=1)
        return self.classifier(fused)
