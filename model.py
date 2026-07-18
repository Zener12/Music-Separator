import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing import N_FFT, SEGMENT_FRAMES
from data_loader import SOURCES

N_SOURCES = len(SOURCES)
FREQ_BINS = N_FFT // 2 + 1


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.down = ConvBlock(out_ch, out_ch, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        x = self.down(skip)
        return (x, skip)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch + out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = x[:, :, :skip.shape[2], :skip.shape[3]]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SourceSeparationNet(nn.Module):
    """U-Net encoder-decoder with multiplicative masking. Input: (B, 1, F, T). Output: (B, n_sources, F, T) masks in [0, 1]."""

    def __init__(self, n_sources: int = N_SOURCES):
        super().__init__()
        self.n_sources = n_sources

        self.encoder = nn.ModuleList([
            EncoderBlock(1, 16),
            EncoderBlock(16, 32),
            EncoderBlock(32, 64)
        ])
        self.bottleneck = ConvBlock(64, 64)
        self.decoder = nn.ModuleList([
            DecoderBlock(64, 32),
            DecoderBlock(32, 16),
            DecoderBlock(16, 16),
        ])
        self.final = nn.Conv2d(16, n_sources, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.encoder[0](x)
        x, skip2 = self.encoder[1](x)
        x, skip3 = self.encoder[2](x)
        x = self.bottleneck(x)
        x = self.decoder[0](x, skip3)
        x = self.decoder[1](x, skip2)
        x = self.decoder[2](x, skip1)
        x = self.final(x)
        x = torch.sigmoid(x)
        return x


def apply_masks(mixture: torch.Tensor, masks: torch.Tensor):
    """mixture: (B, 1, F, T), masks: (B, n_sources, F, T) → (B, n_sources, F, T)"""
    return mixture * masks


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    x = torch.randn(2, 1, FREQ_BINS, SEGMENT_FRAMES)
    print(x.shape)
    model = SourceSeparationNet()
    output = model(x)
    print(output.shape)
    print(f"min = {output.min()}; max = {output.max()}")
    print(count_parameters(model))
