from models.baseline_cnn import BaselineCNN
from models.unet_smp import UNetSMP


def get_model(cfg):
    if cfg.model_name == "baseline_cnn":
        return BaselineCNN(
            dropout=cfg.dropout,
            use_batchnorm=cfg.use_batchnorm,
            activation=cfg.activation,
        )
    if cfg.model_name == "unet_smp":
        return UNetSMP(encoder_name=cfg.encoder_name, pretrained=cfg.pretrained)
    raise ValueError(f"Unknown model_name: {cfg.model_name}")
