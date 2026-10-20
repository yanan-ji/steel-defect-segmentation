"""Train-only, class-aware image sampling informed by multilabel patch flags."""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from .labels import decode_multilabel
from .tiling import starts


def build_patch_index(image_ids: Sequence[str], labels: Mapping[str, Mapping[int, object]],
                      height=256, width=1600, patch_width=512) -> dict:
    """Inspect labels only, using E0's four deterministic audit tiles per image.

    The index estimates which source images can produce minority-class patches;
    training still applies E0's random 256x512 crop to sampled source images.
    """
    lefts = starts(width, patch_width, patch_width)
    flags = np.zeros((len(image_ids), len(lefts), 4), dtype=bool)
    normal_source = np.zeros(len(image_ids), dtype=bool)
    for i, image_id in enumerate(image_ids):
        mask = decode_multilabel(labels.get(image_id, {}), height, width)
        normal_source[i] = not mask.any()
        for j, left in enumerate(lefts):
            flags[i, j] = mask[:, :, left:left + patch_width].any(axis=(1, 2))
    return {"image_ids": list(image_ids), "lefts": lefts, "flags": flags,
            "normal_source": normal_source}


def index_statistics(index: dict) -> dict:
    flags = index["flags"]
    present = flags.any(axis=2)
    return {"source_images": len(index["image_ids"]), "candidate_patches": int(present.size),
            "background_patch_ratio": float((~present).mean()),
            "class_patch_ratio": flags.mean(axis=(0, 1)).tolist(),
            "normal_source_ratio": float(index["normal_source"].mean()),
            "positive_patch_count": int(present.sum())}


def image_weights(index: dict, class_patch_frequency: Sequence[float], cap=3.0) -> np.ndarray:
    """Add every present label; inverse-square-root frequency favors rare classes."""
    frequency = np.asarray(class_patch_frequency, dtype=np.float64)
    if frequency.shape != (4,) or np.any(frequency < 0):
        raise ValueError("Expected four nonnegative class patch frequencies")
    inverse = np.divide(1.0, np.sqrt(frequency), out=np.zeros(4), where=frequency > 0)
    score = (index["flags"].mean(axis=1) * inverse).sum(axis=1)
    nonzero = score[score > 0]
    scale = float(nonzero.mean()) if nonzero.size else 1.0
    return np.clip(1.0 + score / scale, 1.0, cap)


def draw_epoch(index: dict, class_patch_frequency: Sequence[float], seed: int, epoch: int,
               replacement_fraction=.25, max_uses=3) -> np.ndarray:
    """Draw one E0-sized epoch, preserving normal-source and tile-positive strata.

    At most a fraction of each non-normal stratum is exchanged. An image can
    appear at most ``max_uses`` times; each occurrence gets a new E0 random crop.
    """
    if not 0 <= replacement_fraction < 1 or max_uses < 1:
        raise ValueError("Invalid replacement fraction or max uses")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    weights = image_weights(index, class_patch_frequency)
    flags = index["flags"]
    normal_source = np.asarray(index["normal_source"], dtype=bool)
    positive_tile_count = flags.any(axis=2).sum(axis=1)
    selected = []
    for source_normal in (True, False):
        for count in range(flags.shape[1] + 1):
            group = np.flatnonzero((normal_source == source_normal) & (positive_tile_count == count))
            if group.size == 0:
                continue
            # Normal-source images are each used once, retaining E0's source
            # mix exactly; non-normal exchanges are within positivity strata.
            exchange = 0 if source_normal or np.ptp(weights[group]) < 1e-12 else int(np.floor(replacement_fraction * group.size))
            if exchange == 0:
                selected.extend(group.tolist())
                continue
            inverse = 1.0 / weights[group]
            drop = rng.choice(group, size=exchange, replace=False, p=inverse / inverse.sum())
            kept = group[~np.isin(group, drop)]
            uses = {int(i): 1 for i in kept}
            additions = []
            for _ in range(exchange):
                available = np.array([i for i in kept if uses[int(i)] < max_uses], dtype=np.int64)
                if not len(available):
                    raise RuntimeError("Sampling capacity exhausted")
                probabilities = weights[available] / weights[available].sum()
                chosen = int(rng.choice(available, p=probabilities))
                uses[chosen] += 1
                additions.append(chosen)
            selected.extend(kept.tolist() + additions)
    output = np.asarray(selected, dtype=np.int64)
    if len(output) != len(index["image_ids"]) or (np.bincount(output, minlength=len(index["image_ids"])) > max_uses).any():
        raise RuntimeError("Sampling count/cap invariant violated")
    rng.shuffle(output)
    return output


def draw_statistics(index: dict, indices: Sequence[int]) -> dict:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(index["image_ids"])):
        raise ValueError("Invalid sampled source index")
    sampled_flags = index["flags"][indices]
    counts = Counter(indices.tolist())
    return {"draws": len(indices), "unique_source_images": len(counts),
            "duplicate_draws": len(indices) - len(counts), "max_uses_of_one_image": max(counts.values()),
            "normal_source_ratio": float(index["normal_source"][indices].mean()),
            "candidate_background_patch_ratio": float((~sampled_flags.any(axis=2)).mean()),
            "candidate_class_patch_ratio": sampled_flags.mean(axis=(0, 1)).tolist()}
