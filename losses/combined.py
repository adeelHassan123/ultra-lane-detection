import torch
import torch.nn as nn

from losses.tversky import TverskyLoss


class CombinedLoss(nn.Module):
    def __init__(self, pos_weight: float = 3.0, alpha: float = 0.3, beta: float = 0.7, tversky_weight: float = 0.9):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.register_buffer("fallback_pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.tversky_weight = tversky_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bg_count = (targets <= 0.0).sum().float()
        fg_count = (targets >= 1.0).sum().float().clamp(min=1.0)
        dynamic_pos_weight = torch.tensor(
            [float(bg_count / fg_count)],
            dtype=torch.float32,
            device=logits.device,
        )
        pos_weight = torch.maximum(dynamic_pos_weight, self.fallback_pos_weight.to(logits.device))
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight
        )
        tv = self.tversky(logits, targets)
        return self.tversky_weight * tv + (1.0 - self.tversky_weight) * bce
