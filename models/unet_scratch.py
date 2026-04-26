import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetScratch(nn.Module):
    """
    U-Net built from scratch with skip connections.
    Filters: [64, 128, 256, 512, 1024] — ~31M parameters.

    Input:  [B, 3, H, W]
    Output: [B, 1, H, W] raw logits
    """

    def __init__(self):
        super().__init__()
        filters = [64, 128, 256, 512, 1024]

        # Encoder
        self.enc1 = DoubleConv(3, filters[0])
        self.enc2 = DoubleConv(filters[0], filters[1])
        self.enc3 = DoubleConv(filters[1], filters[2])
        self.enc4 = DoubleConv(filters[2], filters[3])
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(filters[3], filters[4])

        # Decoder
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(filters[4], filters[3])

        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(filters[3], filters[2])

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(filters[2], filters[1])

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(filters[1], filters[0])

        # Head
        self.head = nn.Conv2d(filters[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder + save skip connections
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        # Bottleneck
        b = self.bottleneck(self.pool(s4))

        # Decoder + skip connections
        x = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))

        return self.head(x)
