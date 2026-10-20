"""Original-image tiling and reassembly with tail coverage and overlap averaging.

Severstal images are 256x1600. A 512-wide window with a 384 stride leaves a
128-pixel gap at the right edge, so ``starts`` appends one extra window anchored
at ``length - tile``. Overlapping logits are averaged *before* thresholding,
which is why the stitched result is not sensitive to a defect falling on a
window boundary.
"""
from __future__ import annotations

import numpy as np


def starts(length: int, tile: int, stride: int):
    """Return window origins covering ``[0, length)`` including the far tail."""
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile + 1, stride))
    if origins[-1] != length - tile:
        origins.append(length - tile)
    return origins


def tile_image(image, tile_h: int = 256, tile_w: int = 512, stride_h: int = 256, stride_w: int = 512):
    """Yield ``(top, left, patch)`` for every window position."""
    height, width = image.shape[:2]
    return [
        (top, left, image[top:top + tile_h, left:left + tile_w])
        for top in starts(height, tile_h, stride_h)
        for left in starts(width, tile_w, stride_w)
    ]


def stitch_logits(tiles, positions, height: int, width: int):
    """Average overlapping logit windows back to a full ``[C, H, W]`` map."""
    channels = tiles[0].shape[0]
    total = np.zeros((channels, height, width), np.float32)
    count = np.zeros((1, height, width), np.float32)
    for tile, (top, left) in zip(tiles, positions):
        tile_h, tile_w = tile.shape[-2:]
        total[:, top:top + tile_h, left:left + tile_w] += tile
        count[:, top:top + tile_h, left:left + tile_w] += 1
    return total / np.maximum(count, 1)
