import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()  # ensure float32 — ToTensorV2 may produce uint8
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        total = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2 * intersection + self.smooth) / (total + self.smooth)).mean()
        return 0.5 * bce + 0.5 * dice_loss
