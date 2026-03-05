"""B3: ECG-only ResNet baseline (~2.7M params).

Uses a 1-D ResNet with 8 residual blocks to classify 12-lead ECG signals.
"""

import torch
import torch.nn as nn

from src.model import ResNet1D


class ECGOnlyResNet(nn.Module):
    """ResNet-based classifier operating solely on 12-lead ECG waveforms.

    Architecture:
        ResNet1D (8 blocks) → AdaptiveAvgPool → 256 → Linear(256,128)
        → ReLU → Dropout(0.5) → Linear(128,3)

    Args:
        in_channels: Number of ECG leads.
        num_classes: Number of output classes.
        dropout: Dropout probability for the classification head.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.ecg_encoder = ResNet1D(in_channels=in_channels, num_blocks=8)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            ecg: 12-lead ECG tensor of shape ``(batch, 12, seq_len)``.

        Returns:
            Logits of shape ``(batch, 3)``.
        """
        features = self.ecg_encoder(ecg)  # (batch, 256)
        return self.classifier(features)
