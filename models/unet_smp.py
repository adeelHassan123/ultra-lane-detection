import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetSMP(nn.Module):
    """
    U-Net with a pretrained ResNet-18 encoder from segmentation-models-pytorch.
    Exposes get_encoder_params / get_decoder_params for differential learning rates.

    Input:  [B, 3, H, W]
    Output: [B, 1, H, W] raw logits
    """

    def __init__(self, encoder_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        encoder_weights = "imagenet" if pretrained else None
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_encoder_params(self) -> list:
        return list(self.model.encoder.parameters())

    def get_decoder_params(self) -> list:
        return list(self.model.decoder.parameters()) + list(self.model.segmentation_head.parameters())


def build_unet_smp(cfg) -> UNetSMP:
    return UNetSMP(
        encoder_name=cfg.encoder_name,
        pretrained=cfg.pretrained,
    )
