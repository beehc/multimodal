"""B5: Late fusion baseline with learnable branch weights.

Each modality has its own independent encoder *and* classifier head.  The
final prediction is a weighted average of per-branch probability distributions
where the weights are learnable parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import ResNet1D, TabularEncoder, TextEncoder


class LateFusionModel(nn.Module):
    """Late-fusion model with learnable branch weighting.

    Architecture:
        * Text branch:    TextEncoder → classifier → softmax → 3 probs
        * ECG branch:     ResNet1D   → classifier → softmax → 3 probs
        * Tabular branch: TabularEncoder → classifier → softmax → 3 probs
        * Learnable weights (3 params, softmax-normalized) for weighted average.

    Args:
        num_classes: Number of output classes.
        dropout: Dropout probability for branch classifiers.
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

        self.text_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self.ecg_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self.tabular_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

        # Learnable branch weights (initialized uniformly).
        self.branch_weights = nn.Parameter(torch.ones(3))

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
            Combined probability tensor of shape ``(batch, 3)``.  Returns
            probabilities (not raw logits) since the fusion is performed in
            probability space.

        Note:
            Use ``NLLLoss`` with ``torch.log()`` or a custom loss rather
            than ``CrossEntropyLoss`` (which expects raw logits).
        """
        text_features = self.text_encoder(input_ids, attention_mask)
        ecg_features = self.ecg_encoder(ecg)
        tabular_features = self.tabular_encoder(tabular)

        text_probs = F.softmax(self.text_classifier(text_features), dim=1)
        ecg_probs = F.softmax(self.ecg_classifier(ecg_features), dim=1)
        tabular_probs = F.softmax(self.tabular_classifier(tabular_features), dim=1)

        # Normalize branch weights via softmax.
        w = F.softmax(self.branch_weights, dim=0)  # (3,)

        combined = (
            w[0] * text_probs + w[1] * ecg_probs + w[2] * tabular_probs
        )
        return combined
