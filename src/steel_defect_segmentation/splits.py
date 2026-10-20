"""Deterministic image-level splits. Call this before producing any patches."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np

def make_image_splits(image_ids: Iterable[str], seed=42, train=.70, val=.15, test=.15) -> dict[str, list[str]]:
    if not np.isclose(train + val + test, 1): raise ValueError("Fractions must sum to one")
    ids = np.array(sorted(set(image_ids)), dtype=object)
    if not len(ids): raise ValueError("No image IDs")
    np.random.default_rng(seed).shuffle(ids)
    train_end, val_end = int(len(ids) * train), int(len(ids) * (train + val))
    result = {"train": ids[:train_end].tolist(), "val": ids[train_end:val_end].tolist(), "test": ids[val_end:].tolist()}
    if any(set(result[a]) & set(result[b]) for a, b in (("train","val"),("train","test"),("val","test"))): raise RuntimeError("Split leakage")
    return result

def save_splits(splits: dict[str, list[str]], directory: str | Path) -> None:
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items(): (directory / f"{name}_image_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

def make_stratified_image_splits(labels: dict[str, dict[int, object]], seed=42, train=.70, val=.15, test=.15) -> dict[str, list[str]]:
    """Approximate multi-label stratification by exact 4-bit defect-pattern strata."""
    rng=np.random.default_rng(seed); groups={}
    for image_id, rows in labels.items():
        key=tuple(int(bool(isinstance(rows.get(c),str) and rows[c].strip())) for c in range(1,5)); groups.setdefault(key,[]).append(image_id)
    result={"train":[],"val":[],"test":[]}
    for ids in groups.values():
        ids=np.array(ids,dtype=object); rng.shuffle(ids); n=len(ids); a=int(n*train); b=a+int(n*val); result["train"].extend(ids[:a]); result["val"].extend(ids[a:b]); result["test"].extend(ids[b:])
    for ids in result.values(): rng.shuffle(ids)
    return result
