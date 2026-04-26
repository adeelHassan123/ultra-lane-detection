import torch.nn as nn

from models.baseline_cnn import BaselineCNN
from models.unet_scratch import UNetScratch
from models.unet_smp import build_unet_smp


def build_model(cfg) -> nn.Module:
    if cfg.model_name == "baseline_cnn":
        model = BaselineCNN(
            dropout=cfg.dropout,
            use_batchnorm=cfg.use_batchnorm,
            activation=cfg.activation,
        )
    elif cfg.model_name == "unet_scratch":
        model = UNetScratch()
    elif cfg.model_name == "unet_smp":
        model = build_unet_smp(cfg)
    else:
        raise ValueError(
            f"Unknown model_name '{cfg.model_name}'. "
            "Choose from: baseline_cnn, unet_scratch, unet_smp."
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {cfg.model_name} | trainable params: {n_params:,}")
    return model
