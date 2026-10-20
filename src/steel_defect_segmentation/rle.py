"""Severstal RLE helpers: one-indexed, column-major (Fortran) flattened masks."""
from __future__ import annotations
from typing import Optional
import numpy as np

def decode_rle(rle: Optional[str], height: int, width: int) -> np.ndarray:
    """Decode RLE to a uint8 [H,W] mask; missing/empty RLE is all-zero."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    flat = np.zeros(height * width, dtype=np.uint8)
    if rle is None or not str(rle).strip() or str(rle).lower() == "nan":
        return flat.reshape((height, width), order="F")
    values = np.fromstring(str(rle), sep=" ", dtype=np.int64)
    if not len(values) or len(values) % 2:
        raise ValueError("RLE must contain start/length pairs")
    starts, lengths = values[::2] - 1, values[1::2]
    ends = starts + lengths
    if np.any(starts < 0) or np.any(lengths <= 0) or np.any(ends > flat.size):
        raise ValueError("Invalid or out-of-bounds RLE")
    for start, end in zip(starts, ends):
        flat[start:end] = 1
    return flat.reshape((height, width), order="F")

def encode_rle(mask: np.ndarray) -> str:
    """Encode a binary 2D mask to the competition's Fortran-order RLE."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got {mask.shape}")
    pixels = (np.asarray(mask) > 0).astype(np.uint8).reshape(-1, order="F")
    changes = np.flatnonzero(np.diff(np.r_[0, pixels, 0])) + 1
    return " ".join(str(x) for pair in zip(changes[::2], changes[1::2] - changes[::2]) for x in pair)
