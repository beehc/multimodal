"""B2: Text-only Bio-ClinicalBERT baseline (~110.5M params).

Encodes clinical notes with Bio-ClinicalBERT (first 9 layers frozen) and
classifies using the [CLS] representation.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class TextOnlyBERT(nn.Module):
    """BERT-based classifier using only clinical text.

    Architecture:
        Bio-ClinicalBERT (layers 0-8 frozen) → CLS → Linear(768,256)
        → ReLU → Dropout(0.5) → Linear(256,3)

    Args:
        bert_model_name: HuggingFace model identifier.
        freeze_layers: Number of encoder layers to freeze (from the bottom).
        num_classes: Number of output classes.
        dropout: Dropout probability for the classification head.
    """

    def __init__(
        self,
        bert_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        freeze_layers: int = 9,
        num_classes: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)

        # Freeze embeddings and the first `freeze_layers` encoder layers.
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)``.
            attention_mask: Attention mask of shape ``(batch, seq_len)``.

        Returns:
            Logits of shape ``(batch, 3)``.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        return self.classifier(cls_output)
