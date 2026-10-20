"""Losses for four-channel multi-label steel defect segmentation."""
from __future__ import annotations

import torch
from torch import nn


class SoftDiceLoss(nn.Module):
    """Macro Soft Dice over classes with at least one positive target pixel.

    Dice is computed per class across the complete batch using sigmoid(logits).
    A class absent from the current batch is excluded, rather than assigned a
    perfect or zero Dice. An entirely empty batch returns a differentiable zero;
    BCEWithLogitsLoss still penalizes false positives on those batches.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        if smooth <= 0:
            raise ValueError("smooth must be positive")
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape != target.shape or logits.shape[1] != 4:
            raise ValueError("logits and target must both have shape [B,4,H,W]")
        if not target.is_floating_point():
            raise TypeError("target must be floating point")
        probabilities = torch.sigmoid(logits)
        dimensions = (0, 2, 3)
        positives = target.sum(dimensions)
        present = positives > 0
        if not bool(present.any()):
            return logits.sum() * 0.0
        intersection = (probabilities * target).sum(dimensions)
        denominator = probabilities.sum(dimensions) + positives
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return (1.0 - dice[present]).mean()


class BCEDiceLoss(nn.Module):
    """Raw BCEWithLogitsLoss + dice_weight * SoftDiceLoss."""

    def __init__(self, dice_weight: float = 1.0):
        super().__init__()
        if dice_weight < 0:
            raise ValueError("dice_weight must be nonnegative")
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.dice_weight = dice_weight

    def components(self, logits: torch.Tensor, target: torch.Tensor):
        bce = self.bce(logits, target)
        dice = self.dice(logits, target)
        return bce, dice, bce + self.dice_weight * dice

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.components(logits, target)[2]
