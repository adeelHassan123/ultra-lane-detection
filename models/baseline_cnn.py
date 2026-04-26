import torch
import torch.nn as nn
from models.activations import get_activation


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_batchnorm: bool, activation: nn.Module):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(activation)
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    Simple 4-block encoder-decoder CNN. ~200K parameters.
    Intentionally weak — this is the floor for all comparisons.

    Input:  [B, 3, H, W]
    Output: [B, 1, H, W] raw logits
    """

    def __init__(
        self,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        activation: str = "relu",
    ):
        super().__init__()

        def _act():
            return get_activation(activation)

        # Encoder
        self.enc1 = ConvBlock(3, 32, use_batchnorm, _act())
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64, use_batchnorm, _act())
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(64, 128, use_batchnorm, _act())
        self.dropout = nn.Dropout2d(p=dropout)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(128, 64, use_batchnorm, _act())

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(64, 32, use_batchnorm, _act())

        # Head
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.dropout(self.bottleneck(x))
        x = self.dec1(self.up1(x))
        x = self.dec2(self.up2(x))
        return self.head(x)
