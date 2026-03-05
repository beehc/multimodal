"""Loss functions for the Multi-Modal Emergency Triage System."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in triage classification.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter that down-weights easy examples.
        alpha: Class balancing factor.
        reduction: Reduction method ('mean' or 'sum').
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"Invalid reduction '{reduction}'. Expected 'mean', 'sum', or 'none'."
            )
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Raw logits of shape (batch_size, num_classes).
            targets: Class indices of shape (batch_size,).

        Returns:
            Scalar focal loss tensor.
        """
        # Per-element CE so focal weights can be applied before reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining a main loss with auxiliary losses (v7 optimization).

    Args:
        main_loss_fn: Primary loss function (e.g., FocalLoss).
        auxiliary_weight: Weight applied to auxiliary losses (alpha = 0.1).
    """

    def __init__(
        self,
        main_loss_fn: nn.Module,
        auxiliary_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.main_loss_fn = main_loss_fn
        self.auxiliary_weight = auxiliary_weight

    def forward(
        self,
        main_logits: torch.Tensor,
        targets: torch.Tensor,
        aux_logits_list: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute multi-task loss.

        Args:
            main_logits: Logits from the main classification head.
            targets: Ground-truth class indices.
            aux_logits_list: Optional list of logits from auxiliary heads.

        Returns:
            Tuple of (total_loss, info_dict) where info_dict contains
            'main_loss' and 'aux_loss' values for logging.
        """
        main_loss = self.main_loss_fn(main_logits, targets)
        aux_loss = torch.tensor(0.0, device=main_loss.device)

        if aux_logits_list is not None:
            for aux_logits in aux_logits_list:
                aux_loss = aux_loss + F.cross_entropy(aux_logits, targets)
            total_loss = main_loss + self.auxiliary_weight * aux_loss
        else:
            total_loss = main_loss

        return total_loss, {"main_loss": main_loss, "aux_loss": aux_loss}
