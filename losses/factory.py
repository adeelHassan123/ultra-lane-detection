import torch
import torch.nn as nn

from losses.tversky_loss import TverskyLoss
from losses.dice_loss import DiceLoss
from losses.combined_loss import CombinedLoss


def build_loss(cfg) -> nn.Module:
    if cfg.loss_name == "bce":
        pos_weight = torch.tensor([cfg.positive_weight])
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if cfg.loss_name == "dice":
        return DiceLoss(smooth=cfg.tversky_smooth)

    if cfg.loss_name == "tversky":
        return TverskyLoss(
            alpha=cfg.tversky_alpha,
            beta=cfg.tversky_beta,
            smooth=cfg.tversky_smooth,
        )

    if cfg.loss_name == "combined":
        pos_weight = torch.tensor([cfg.positive_weight])
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        tversky = TverskyLoss(
            alpha=cfg.tversky_alpha,
            beta=cfg.tversky_beta,
            smooth=cfg.tversky_smooth,
        )
        return CombinedLoss(
            tversky_weight=cfg.tversky_weight,
            bce_loss=bce,
            tversky_loss=tversky,
        )

    raise ValueError(f"Unknown loss_name '{cfg.loss_name}'. Choose from: bce, dice, tversky, combined.")
