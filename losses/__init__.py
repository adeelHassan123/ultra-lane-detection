from losses.combined import CombinedLoss
from losses.dice__bce import DiceBCELoss
from losses.tversky import TverskyLoss


def get_loss(cfg):
    if cfg.loss_name == "combined":
        return CombinedLoss(
            pos_weight=cfg.positive_weight,
            alpha=cfg.tversky_alpha,
            beta=cfg.tversky_beta,
            tversky_weight=cfg.tversky_weight,
        )
    if cfg.loss_name == "tversky":
        return TverskyLoss(alpha=cfg.tversky_alpha, beta=cfg.tversky_beta, smooth=cfg.tversky_smooth)
    if cfg.loss_name == "dice_bce":
        return DiceBCELoss()
    if cfg.loss_name == "bce":
        return __import__("torch").nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown loss_name: {cfg.loss_name}")
