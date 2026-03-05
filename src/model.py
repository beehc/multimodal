"""Multi-Modal Emergency Triage System — model definitions.

This module contains all encoder, fusion, and classifier components for the
multi-modal triage pipeline (v6 and v7 architectures).
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ECG Encoder Components
# ---------------------------------------------------------------------------


class ResidualBlock1D(nn.Module):
    """1-D residual block with two convolutions, batch-norm, and ReLU.

    A shortcut (1×1 conv) is added when the spatial dimensions or channel
    counts change between input and output.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size (default 15).
        stride: Stride for the *first* convolution (default 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection when dimensions change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, channels, length)``.

        Returns:
            Output tensor of shape ``(batch, out_channels, length')``.
        """
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNet1D(nn.Module):
    """1-D ResNet backbone for ECG signals.

    Architecture:
        * Initial conv: ``in_channels`` → ``base_channels``, kernel 15, stride 2
        * 8 residual blocks in 4 groups:
          - Blocks 1-2: 64 → 64,  stride 1
          - Blocks 3-4: 64 → 128, stride 2
          - Blocks 5-6: 128 → 256, stride 2
          - Blocks 7-8: 256 → 256, stride 2
        * AdaptiveAvgPool1d(1) + flatten → 256-d feature vector

    Args:
        in_channels: Number of ECG leads (default 12).
        base_channels: Channels after the initial convolution (default 64).
        kernel_size: Kernel size for all residual blocks (default 15).
        num_blocks: Total number of residual blocks (default 8).
    """

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        kernel_size: int = 15,
        num_blocks: int = 8,
    ) -> None:
        super().__init__()
        _ = num_blocks  # kept for API compatibility

        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv1d(
                in_channels, base_channels, kernel_size=15, stride=2,
                padding=7, bias=False,
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Group 1: blocks 1-2 (64 → 64, stride 1)
        self.block1 = ResidualBlock1D(64, 64, kernel_size, stride=1)
        self.block2 = ResidualBlock1D(64, 64, kernel_size, stride=1)

        # Group 2: blocks 3-4 (64 → 128, stride 2 on first)
        self.block3 = ResidualBlock1D(64, 128, kernel_size, stride=2)
        self.block4 = ResidualBlock1D(128, 128, kernel_size, stride=1)

        # Group 3: blocks 5-6 (128 → 256, stride 2 on first)
        self.block5 = ResidualBlock1D(128, 256, kernel_size, stride=2)
        self.block6 = ResidualBlock1D(256, 256, kernel_size, stride=1)

        # Group 4: blocks 7-8 (256 → 256, stride 2 on first)
        self.block7 = ResidualBlock1D(256, 256, kernel_size, stride=2)
        self.block8 = ResidualBlock1D(256, 256, kernel_size, stride=1)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ECG tensor of shape ``(batch, 12, 5000)``.

        Returns:
            Feature vector of shape ``(batch, 256)``.
        """
        x = self.init_conv(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        x = self.pool(x)          # (batch, 256, 1)
        x = x.flatten(start_dim=1)  # (batch, 256)
        return x


class TransformerEncoderLayer1D(nn.Module):
    """Simple transformer encoder layer for 1-D sequences (v7).

    Implements multi-head self-attention followed by a position-wise FFN,
    each wrapped with residual connections and layer normalisation.

    Args:
        d_model: Embedding / feature dimension (default 256).
        nhead: Number of attention heads (default 4).
        dim_feedforward: Hidden dimension of the FFN (default 512).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Output tensor of the same shape.
        """
        # Self-attention with residual + norm
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual + norm
        x = self.norm2(x + self.ffn(x))
        return x


class HybridECGEncoder(nn.Module):
    """Hybrid ECG encoder combining ResNet, Transformer, and attentive pooling (v7).

    Pipeline:
        1. **ResNet stage** — 4 residual blocks (64→64→128→256), kernel 15.
        2. **Positional embeddings** — learnable 1-D embeddings added to the
           feature sequence.
        3. **Transformer stage** — 2-layer transformer with 4 heads.
        4. **Attentive pooling** — learned attention weights collapse the
           temporal dimension.

    Input shape:  ``(batch, 12, 5000)``
    Output shape: ``(batch, 256)``
    """

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        kernel_size: int = 15,
    ) -> None:
        super().__init__()

        # Stage 1: ResNet with 4 blocks
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.res_block1 = ResidualBlock1D(64, 64, kernel_size, stride=1)
        self.res_block2 = ResidualBlock1D(64, 64, kernel_size, stride=2)
        self.res_block3 = ResidualBlock1D(64, 128, kernel_size, stride=2)
        self.res_block4 = ResidualBlock1D(128, 256, kernel_size, stride=2)

        # Stage 2: Learnable positional embeddings (registered lazily)
        self._pos_emb_initialized = False
        self.pos_embedding: nn.Parameter  # set in _init_pos_embedding

        # Stage 3: 2-layer Transformer
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer1D(d_model=256, nhead=4, dim_feedforward=512, dropout=0.1)
            for _ in range(2)
        ])

        # Stage 4: Attentive pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(256, 1),
        )

    def _init_pos_embedding(self, seq_len: int, device: torch.device) -> None:
        """Lazily initialise positional embeddings to match the sequence length."""
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, 256, device=device) * 0.02)
        self._pos_emb_initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ECG tensor of shape ``(batch, 12, 5000)``.

        Returns:
            Feature vector of shape ``(batch, 256)``.
        """
        # Stage 1: ResNet feature extraction
        x = self.init_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)  # (batch, 256, T)

        # Transpose to (batch, T, 256) for transformer
        x = x.permute(0, 2, 1)
        seq_len = x.size(1)

        # Stage 2: Positional embeddings
        if not self._pos_emb_initialized:
            self._init_pos_embedding(seq_len, x.device)
        if self.pos_embedding.size(1) != seq_len:
            self._init_pos_embedding(seq_len, x.device)
        x = x + self.pos_embedding[:, :seq_len, :]

        # Stage 3: Transformer
        for layer in self.transformer_layers:
            x = layer(x)  # (batch, T, 256)

        # Stage 4: Attentive pooling
        attn_weights = torch.softmax(self.attn_pool(x), dim=1)  # (batch, T, 1)
        x = (x * attn_weights).sum(dim=1)  # (batch, 256)
        return x


# ---------------------------------------------------------------------------
# Text Encoder Components
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    """BERT-based clinical text encoder (v6).

    Loads a pretrained BERT model, freezes the embedding layer and the first
    ``freeze_layers`` transformer layers, and projects the CLS token
    representation to a 256-d feature vector.

    Args:
        bert_model_name: HuggingFace model identifier.
        freeze_layers: Number of transformer layers to freeze (default 9).
    """

    def __init__(
        self,
        bert_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        freeze_layers: int = 9,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)

        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze first `freeze_layers` transformer layers
        for i in range(freeze_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

        self.projection = nn.Linear(768, 256)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token ids of shape ``(batch, seq_len)``.
            attention_mask: Attention mask of shape ``(batch, seq_len)``.

        Returns:
            Feature vector of shape ``(batch, 256)``.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        return self.projection(cls_token)


class MultiGranularityTextEncoder(nn.Module):
    """Multi-granularity clinical text encoder (v7).

    Extracts both the CLS token (768-d) and all token embeddings (seq_len×768).
    A learned attention mechanism computes a weighted sum over the token
    representations. The CLS embedding and the weighted token embedding are
    concatenated and projected to 256 dimensions.

    Args:
        bert_model_name: HuggingFace model identifier.
        freeze_layers: Number of transformer layers to freeze (default 9).
    """

    def __init__(
        self,
        bert_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        freeze_layers: int = 9,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)

        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze first `freeze_layers` transformer layers
        for i in range(freeze_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

        # Learnable attention weights for token-level aggregation
        self.token_attention = nn.Sequential(
            nn.Linear(768, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        # Projection: CLS (768) + weighted tokens (768) → 256
        self.projection = nn.Linear(768 * 2, 256)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token ids of shape ``(batch, seq_len)``.
            attention_mask: Attention mask of shape ``(batch, seq_len)``.

        Returns:
            Feature vector of shape ``(batch, 256)``.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, 768)

        cls_token = hidden_states[:, 0, :]  # (batch, 768)

        # Token-level attention (exclude CLS at index 0)
        token_embeddings = hidden_states[:, 1:, :]  # (batch, seq_len-1, 768)
        token_mask = attention_mask[:, 1:].unsqueeze(-1)  # (batch, seq_len-1, 1)

        attn_scores = self.token_attention(token_embeddings)  # (batch, seq_len-1, 1)
        # Mask padding tokens with large negative value
        attn_scores = attn_scores.masked_fill(token_mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len-1, 1)

        weighted_tokens = (token_embeddings * attn_weights).sum(dim=1)  # (batch, 768)

        # Concatenate and project
        combined = torch.cat([cls_token, weighted_tokens], dim=-1)  # (batch, 1536)
        return self.projection(combined)  # (batch, 256)


# ---------------------------------------------------------------------------
# Tabular Encoder Components
# ---------------------------------------------------------------------------


class TabularEncoder(nn.Module):
    """Three-layer MLP encoder for tabular vitals / demographics (v6).

    Architecture: 9 → 64 → 64 → 64 with ReLU, BatchNorm, and Dropout.

    Args:
        input_dim: Number of tabular features (default 9).
        hidden_dim: Hidden layer width (default 64).
        dropout: Dropout probability (default 0.3).
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tabular features of shape ``(batch, 9)``.

        Returns:
            Feature vector of shape ``(batch, 64)``.
        """
        return self.mlp(x)


class FeatureInteractionLayer(nn.Module):
    """Pairwise feature interaction layer (v7).

    Given *n* input features, produces the original *n* features concatenated
    with all ``n*(n-1)/2`` pairwise element-wise products.

    For ``n = 9``: output = 9 + 36 = 45 features.

    Args:
        input_dim: Number of raw tabular features (default 9).
    """

    def __init__(self, input_dim: int = 9) -> None:
        super().__init__()
        self.input_dim = input_dim
        # Pre-compute pair indices
        pairs_i: list[int] = []
        pairs_j: list[int] = []
        for i in range(input_dim):
            for j in range(i + 1, input_dim):
                pairs_i.append(i)
                pairs_j.append(j)
        self.register_buffer("pairs_i", torch.tensor(pairs_i, dtype=torch.long))
        self.register_buffer("pairs_j", torch.tensor(pairs_j, dtype=torch.long))
        self.output_dim = input_dim + len(pairs_i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tabular features of shape ``(batch, input_dim)``.

        Returns:
            Augmented features of shape ``(batch, input_dim + n_pairs)``.
        """
        interactions = x[:, self.pairs_i] * x[:, self.pairs_j]  # (batch, n_pairs)
        return torch.cat([x, interactions], dim=-1)


class EnhancedTabularEncoder(nn.Module):
    """Enhanced tabular encoder with feature interactions and residual MLP (v7).

    Pipeline: :class:`FeatureInteractionLayer` → deep MLP with residual
    connection.  45 → 128 → 128 → 128.

    Output: 128-d feature vector.

    Args:
        input_dim: Number of raw tabular features (default 9).
        hidden_dim: Hidden layer width (default 128).
        dropout: Dropout probability (default 0.3).
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.interaction = FeatureInteractionLayer(input_dim)
        interaction_dim = self.interaction.output_dim  # 45 for input_dim=9

        self.input_proj = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Residual block: 128 → 128
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tabular features of shape ``(batch, 9)``.

        Returns:
            Feature vector of shape ``(batch, 128)``.
        """
        x = self.interaction(x)   # (batch, 45)
        x = self.input_proj(x)    # (batch, 128)
        x = self.relu(x + self.res_block(x))  # residual connection
        return x


# ---------------------------------------------------------------------------
# Fusion Components
# ---------------------------------------------------------------------------


class CrossAttentionModule(nn.Module):
    """Bidirectional cross-attention between text and ECG features.

    Two attention paths:
        * **Path A** — text queries ECG: Q = text, K = ECG, V = ECG
        * **Path B** — ECG queries text: Q = ECG, K = text, V = text

    Both paths use 8 heads with head_dim = 32 (total = 256).  The outputs of
    the two paths are concatenated to produce a 512-d fused representation.

    Args:
        embed_dim: Feature dimension for each modality (default 256).
        num_heads: Number of attention heads (default 8).
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8) -> None:
        super().__init__()
        self.attn_text_to_ecg = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True,
        )
        self.attn_ecg_to_text = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True,
        )

    def forward(
        self,
        text_features: torch.Tensor,
        ecg_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            text_features: Shape ``(batch, 256)``.
            ecg_features: Shape ``(batch, 256)``.

        Returns:
            Fused features of shape ``(batch, 512)``.
        """
        # Unsqueeze to create seq_len=1 for multihead attention
        text_seq = text_features.unsqueeze(1)  # (batch, 1, 256)
        ecg_seq = ecg_features.unsqueeze(1)    # (batch, 1, 256)

        # Path A: text queries ECG
        attn_a, _ = self.attn_text_to_ecg(
            query=text_seq, key=ecg_seq, value=ecg_seq,
        )  # (batch, 1, 256)

        # Path B: ECG queries text
        attn_b, _ = self.attn_ecg_to_text(
            query=ecg_seq, key=text_seq, value=text_seq,
        )  # (batch, 1, 256)

        # Squeeze and concatenate
        attn_a = attn_a.squeeze(1)  # (batch, 256)
        attn_b = attn_b.squeeze(1)  # (batch, 256)
        return torch.cat([attn_a, attn_b], dim=-1)  # (batch, 512)


class TabularConditionedCrossAttention(nn.Module):
    """Cross-attention with tabular-conditioned query modulation (v7).

    Similar to :class:`CrossAttentionModule` but the query vectors are
    element-wise scaled by sigmoid-activated projections of the tabular
    features before being fed into the cross-attention heads.

    Args:
        embed_dim: Feature dimension for text / ECG (default 256).
        num_heads: Number of attention heads (default 8).
        tabular_dim: Dimension of tabular features (default 128).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        tabular_dim: int = 128,
    ) -> None:
        super().__init__()
        self.attn_text_to_ecg = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True,
        )
        self.attn_ecg_to_text = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True,
        )

        # Tabular conditioning: generate scale factors for queries
        self.text_condition = nn.Sequential(
            nn.Linear(tabular_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.ecg_condition = nn.Sequential(
            nn.Linear(tabular_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        ecg_features: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            text_features: Shape ``(batch, 256)``.
            ecg_features: Shape ``(batch, 256)``.
            tabular_features: Shape ``(batch, tabular_dim)``.

        Returns:
            Fused features of shape ``(batch, 512)``.
        """
        # Generate modulation factors from tabular features
        text_scale = self.text_condition(tabular_features)  # (batch, 256)
        ecg_scale = self.ecg_condition(tabular_features)    # (batch, 256)

        # Modulate queries
        text_query = (text_features * text_scale).unsqueeze(1)  # (batch, 1, 256)
        ecg_query = (ecg_features * ecg_scale).unsqueeze(1)     # (batch, 1, 256)

        ecg_seq = ecg_features.unsqueeze(1)   # (batch, 1, 256)
        text_seq = text_features.unsqueeze(1)  # (batch, 1, 256)

        # Path A: modulated text queries ECG
        attn_a, _ = self.attn_text_to_ecg(
            query=text_query, key=ecg_seq, value=ecg_seq,
        )

        # Path B: modulated ECG queries text
        attn_b, _ = self.attn_ecg_to_text(
            query=ecg_query, key=text_seq, value=text_seq,
        )

        attn_a = attn_a.squeeze(1)  # (batch, 256)
        attn_b = attn_b.squeeze(1)  # (batch, 256)
        return torch.cat([attn_a, attn_b], dim=-1)  # (batch, 512)


class GatedMultimodalUnit(nn.Module):
    """Gated Multimodal Unit for fusing cross-attention output with tabular features (v6).

    Both the fused features (512-d) and tabular features (64-d) are projected
    to 256-d.  A sigmoid gate decides the per-element mixture.  The output is
    the gated combination concatenated with the projected fused features and
    the original tabular features:

        output = cat(gated(256), h_fused(256), tabular(64)) → **576-d**

    Args:
        fused_dim: Dimension of cross-attention output (default 512).
        tabular_dim: Dimension of tabular features (default 64).
        gate_dim: Projection dimension for gating (default 256).
    """

    def __init__(
        self,
        fused_dim: int = 512,
        tabular_dim: int = 64,
        gate_dim: int = 256,
    ) -> None:
        super().__init__()
        self.proj_fused = nn.Linear(fused_dim, gate_dim)
        self.proj_tabular = nn.Linear(tabular_dim, gate_dim)
        self.gate = nn.Linear(gate_dim * 2, gate_dim)

    def forward(
        self,
        fused: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            fused: Cross-attention features of shape ``(batch, 512)``.
            tabular: Tabular features of shape ``(batch, 64)``.

        Returns:
            Combined features of shape ``(batch, 576)``.
        """
        h_fused = self.proj_fused(fused)      # (batch, 256)
        h_tabular = self.proj_tabular(tabular)  # (batch, 256)

        gate_input = torch.cat([h_fused, h_tabular], dim=-1)  # (batch, 512)
        g = torch.sigmoid(self.gate(gate_input))  # (batch, 256)

        gated = g * h_fused + (1 - g) * h_tabular  # (batch, 256)

        # Concatenate gated combination with projected fused + original tabular
        return torch.cat([gated, h_fused, tabular], dim=-1)  # (batch, 576)


class ChannelWiseGMU(nn.Module):
    """Channel-wise Gated Multimodal Unit (v7).

    Uses element-wise (channel-level) gating instead of a global gate.
    Separate gate vectors are produced for the fused and tabular branches,
    enabling fine-grained per-dimension modulation.

    Output dimension = gated(256) + proj_fused(256) + tabular(tabular_dim).

    Args:
        fused_dim: Dimension of cross-attention output (default 512).
        tabular_dim: Dimension of enhanced tabular features (default 128).
        gate_dim: Projection dimension for gating (default 256).
    """

    def __init__(
        self,
        fused_dim: int = 512,
        tabular_dim: int = 128,
        gate_dim: int = 256,
    ) -> None:
        super().__init__()
        self.gate_dim = gate_dim
        self.tabular_dim = tabular_dim

        self.proj_fused = nn.Linear(fused_dim, gate_dim)
        self.proj_tabular = nn.Linear(tabular_dim, gate_dim)

        # Channel-wise gate vectors
        self.gate_fused = nn.Linear(gate_dim * 2, gate_dim)
        self.gate_tabular = nn.Linear(gate_dim * 2, gate_dim)

    @property
    def output_dim(self) -> int:
        """Total output dimensionality."""
        return self.gate_dim + self.gate_dim + self.tabular_dim

    def forward(
        self,
        fused: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            fused: Cross-attention features of shape ``(batch, fused_dim)``.
            tabular: Enhanced tabular features of shape ``(batch, tabular_dim)``.

        Returns:
            Combined features of shape ``(batch, output_dim)``.
        """
        h_fused = self.proj_fused(fused)        # (batch, 256)
        h_tabular = self.proj_tabular(tabular)  # (batch, 256)

        gate_input = torch.cat([h_fused, h_tabular], dim=-1)  # (batch, 512)

        # Element-wise (channel) gates
        g_fused = torch.sigmoid(self.gate_fused(gate_input))      # (batch, 256)
        g_tabular = torch.sigmoid(self.gate_tabular(gate_input))  # (batch, 256)

        gated = g_fused * h_fused + g_tabular * h_tabular  # (batch, 256)

        return torch.cat([gated, h_fused, tabular], dim=-1)  # (batch, output_dim)


# ---------------------------------------------------------------------------
# Main Models
# ---------------------------------------------------------------------------


class MultiModalTriageModel(nn.Module):
    """Multi-modal triage model — v6 architecture.

    Encoders:
        * :class:`TextEncoder` (Bio_ClinicalBERT → 256-d)
        * :class:`ResNet1D` (12-lead ECG → 256-d)
        * :class:`TabularEncoder` (vitals → 64-d)

    Fusion:
        * :class:`CrossAttentionModule` (text + ECG → 512-d)
        * :class:`GatedMultimodalUnit` (fused + tabular → 576-d)

    Classifier: 576 → 256 → 128 → 3 (ReLU, BN, Dropout 0.5/0.3).
    """

    def __init__(self) -> None:
        super().__init__()
        # Encoders
        self.text_encoder = TextEncoder()
        self.ecg_encoder = ResNet1D()
        self.tabular_encoder = TabularEncoder()

        # Fusion
        self.cross_attention = CrossAttentionModule()
        self.gmu = GatedMultimodalUnit()

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 3),
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
            input_ids: Token ids ``(batch, seq_len)``.
            attention_mask: Attention mask ``(batch, seq_len)``.
            ecg: ECG signal ``(batch, 12, 5000)``.
            tabular: Tabular features ``(batch, 9)``.

        Returns:
            Logits of shape ``(batch, 3)``.
        """
        text_feat = self.text_encoder(input_ids, attention_mask)  # (batch, 256)
        ecg_feat = self.ecg_encoder(ecg)                          # (batch, 256)
        tab_feat = self.tabular_encoder(tabular)                  # (batch, 64)

        fused = self.cross_attention(text_feat, ecg_feat)         # (batch, 512)
        combined = self.gmu(fused, tab_feat)                      # (batch, 576)

        return self.classifier(combined)                          # (batch, 3)


class MultiModalTriageModelOptimized(nn.Module):
    """Multi-modal triage model — v7 optimised architecture.

    Encoders:
        * :class:`MultiGranularityTextEncoder` (Bio_ClinicalBERT → 256-d)
        * :class:`HybridECGEncoder` (12-lead ECG → 256-d)
        * :class:`EnhancedTabularEncoder` (vitals → 128-d)

    Fusion:
        * :class:`TabularConditionedCrossAttention` (text + ECG | tabular → 512-d)
        * :class:`ChannelWiseGMU` (fused + tabular → variable-d)

    Classifier head: ``gmu_output_dim`` → 256 → 128 → 3.

    Auxiliary heads (one per modality):
        * text  : 256 → 3
        * ECG   : 256 → 3
        * tabular: 128 → 3

    ``forward`` returns ``(main_logits, [text_aux, ecg_aux, tab_aux])``.
    """

    def __init__(self) -> None:
        super().__init__()
        # Encoders
        self.text_encoder = MultiGranularityTextEncoder()
        self.ecg_encoder = HybridECGEncoder()
        self.tabular_encoder = EnhancedTabularEncoder()

        # Fusion
        self.cross_attention = TabularConditionedCrossAttention(
            embed_dim=256, num_heads=8, tabular_dim=128,
        )
        self.gmu = ChannelWiseGMU(fused_dim=512, tabular_dim=128, gate_dim=256)

        gmu_out = self.gmu.output_dim  # 256 + 256 + 128 = 640

        # Main classifier head
        self.classifier = nn.Sequential(
            nn.Linear(gmu_out, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 3),
        )

        # Auxiliary heads (one per modality)
        self.aux_text = nn.Linear(256, 3)
        self.aux_ecg = nn.Linear(256, 3)
        self.aux_tab = nn.Linear(128, 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ecg: torch.Tensor,
        tabular: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.

        Args:
            input_ids: Token ids ``(batch, seq_len)``.
            attention_mask: Attention mask ``(batch, seq_len)``.
            ecg: ECG signal ``(batch, 12, 5000)``.
            tabular: Tabular features ``(batch, 9)``.

        Returns:
            A tuple of:
                * ``main_logits`` of shape ``(batch, 3)``.
                * A list ``[text_aux, ecg_aux, tab_aux]`` each ``(batch, 3)``.
        """
        text_feat = self.text_encoder(input_ids, attention_mask)  # (batch, 256)
        ecg_feat = self.ecg_encoder(ecg)                          # (batch, 256)
        tab_feat = self.tabular_encoder(tabular)                  # (batch, 128)

        fused = self.cross_attention(text_feat, ecg_feat, tab_feat)  # (batch, 512)
        combined = self.gmu(fused, tab_feat)                         # (batch, gmu_out)

        main_logits = self.classifier(combined)  # (batch, 3)

        # Auxiliary predictions
        text_aux = self.aux_text(text_feat)  # (batch, 3)
        ecg_aux = self.aux_ecg(ecg_feat)     # (batch, 3)
        tab_aux = self.aux_tab(tab_feat)     # (batch, 3)

        return main_logits, [text_aux, ecg_aux, tab_aux]
