"""B1: Tabular-only MLP baseline (~50K params).

Uses only the 9 structured clinical features to predict triage acuity.
"""

import torch
import torch.nn as nn


class TabularOnlyMLP(nn.Module):
    """MLP classifier operating solely on tabular clinical features.

    Architecture: 9 → 128 → 256 → 128 → 3 with BatchNorm, ReLU, and Dropout.

    Args:
        input_dim: Number of tabular input features.
        num_classes: Number of output classes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 9,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, tabular: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            tabular: Tensor of shape ``(batch, 9)``.

        Returns:
            Logits of shape ``(batch, 3)``.
        """
        return self.classifier(tabular)
