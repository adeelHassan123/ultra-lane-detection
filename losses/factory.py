import torch
import torch.nn as nn

from losses.tversky import TverskyLoss
from losses.dice_bce import DiceBCELoss
from losses.combined import CombinedLoss


def build_loss(cfg) -> nn.Module:
    if cfg.loss_name == "bce":
        pos_weight = torch.tensor([cfg.positive_weight])
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if cfg.loss_name == "dice":
        return DiceBCELoss(smooth=cfg.tversky_smooth)

    if cfg.loss_name == "tversky":
        return TverskyLoss(
            alpha=cfg.tversky_alpha,
            beta=cfg.tversky_beta,
            smooth=cfg.tversky_smooth,
        )

    if cfg.loss_name == "combined":
        return CombinedLoss(
            pos_weight=cfg.positive_weight,
            alpha=cfg.tversky_alpha,
            beta=cfg.tversky_beta,
            tversky_weight=cfg.tversky_weight,
        )

    raise ValueError(f"Unknown loss_name '{cfg.loss_name}'. Choose from: bce, dice, tversky, combined.")

