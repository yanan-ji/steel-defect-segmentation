"""ResNet18-style U-Net for four-channel multi-label steel defect segmentation.

The encoder mirrors the ResNet18 stage layout (2 x BasicBlock per stage, channel
widths 64/128/256/512, stride-2 downsampling) but is implemented here without a
torchvision dependency, so the project installs and runs without downloading
ImageNet weights. All weights are trained from scratch on the Severstal data.

Attribute names are part of the public checkpoint contract: every released
``.pt`` / ``.onnx`` artifact was produced by this module layout, and
``load_state_dict(..., strict=True)`` is expected to succeed unchanged.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    """Residual basic block: two 3x3 convolutions plus a projection shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.a = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_ch == out_ch
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.a(x) + self.skip(x), inplace=True)


class Decoder(nn.Module):
    """Bilinear upsample, encoder-skip concatenation, then two 3x3 convolutions."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([upsampled, skip], 1))


class ResNet18UNet(nn.Module):
    """Returns raw ``[B, classes, H, W]`` logits.

    The model deliberately applies no sigmoid: losses use
    ``BCEWithLogitsLoss`` for numerical stability, and metrics apply sigmoid
    only where probabilities are actually needed. The frozen inference policy
    of threshold 0.50 is therefore equivalent to ``logit >= 0``.
    """

    def __init__(self, classes: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.l1 = nn.Sequential(Block(64, 64), Block(64, 64))
        self.l2 = nn.Sequential(Block(64, 128, 2), Block(128, 128))
        self.l3 = nn.Sequential(Block(128, 256, 2), Block(256, 256))
        self.l4 = nn.Sequential(Block(256, 512, 2), Block(512, 512))
        self.d3 = Decoder(512, 256, 256)
        self.d2 = Decoder(256, 128, 128)
        self.d1 = Decoder(128, 64, 64)
        self.out = nn.Conv2d(64, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = self.stem(x)
        a = self.l1(x)
        b = self.l2(a)
        c = self.l3(b)
        d = self.l4(c)
        x = self.d3(d, c)
        x = self.d2(x, b)
        x = self.d1(x, a)
        return F.interpolate(self.out(x), size=size, mode="bilinear", align_corners=False)
