"""Read Severstal labels into four independent binary channels."""
from __future__ import annotations
from pathlib import Path
from typing import Mapping
import numpy as np
import pandas as pd
from .rle import decode_rle

NUM_CLASSES = 4

def read_label_table(csv_path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(csv_path)
    if {"ImageId", "ClassId", "EncodedPixels"}.issubset(table.columns):
        result = table
    elif {"ImageId_ClassId", "EncodedPixels"}.issubset(table.columns):
        parts = table.ImageId_ClassId.str.rsplit("_", n=1, expand=True)
        result = table.assign(ImageId=parts[0], ClassId=parts[1].astype(int))
    else:
        raise ValueError("CSV must use ImageId/ClassId/EncodedPixels or ImageId_ClassId/EncodedPixels")
    if not result.ClassId.between(1, NUM_CLASSES).all(): raise ValueError("ClassId must be 1..4")
    return result[["ImageId", "ClassId", "EncodedPixels"]]

def labels_by_image(table: pd.DataFrame) -> Mapping[str, Mapping[int, object]]:
    return {key: dict(zip(group.ClassId, group.EncodedPixels)) for key, group in table.groupby("ImageId", sort=False)}

def decode_multilabel(rows: Mapping[int, object], height: int, width: int) -> np.ndarray:
    """Return [4,H,W] uint8, preserving simultaneous defect-class labels."""
    return np.stack([decode_rle(rows.get(c), height, width) for c in range(1, NUM_CLASSES + 1)])
