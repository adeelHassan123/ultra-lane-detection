import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """0.9 * TverskyLoss + 0.1 * BCEWithLogitsLoss by default."""

    def __init__(self, tversky_weight: float, bce_loss: nn.Module, tversky_loss: nn.Module):
        super().__init__()
        self.tversky_weight = tversky_weight
        self.bce_loss = bce_loss
        self.tversky_loss = tversky_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.tversky_weight * self.tversky_loss(logits, targets)
            + (1 - self.tversky_weight) * self.bce_loss(logits, targets)
        )
