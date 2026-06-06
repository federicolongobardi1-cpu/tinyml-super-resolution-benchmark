import torch
from torch import nn


class ESPCN(nn.Module):
    def __init__(self, upscale_factor: int, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.sub_pixel = nn.Sequential(
            nn.Conv2d(32, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x):
        x = self.feature_maps(x)
        x = self.sub_pixel(x)
        x = torch.clamp(x, 0.0, 1.0)
        return x