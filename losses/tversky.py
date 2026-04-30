import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        tp = (probs * targets).sum(dim=(1, 2, 3))
        fp = (probs * (1 - targets)).sum(dim=(1, 2, 3))
        fn = ((1 - probs) * targets).sum(dim=(1, 2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()
