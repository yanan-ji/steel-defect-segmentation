#!/usr/bin/env python3
"""Index only frozen training-label tiles for the E2 sampler; no image training."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steel_defect_segmentation.labels import labels_by_image, read_label_table
from steel_defect_segmentation.sampling import build_patch_index, draw_epoch, draw_statistics, index_statistics


TRAIN_HASH = "ba7887219c3d07215f44b704f053dff6bfece91375bd7a5178353af156152c89"
VAL_HASH = "593cf50fe4c05f2d788f3fb8a15a55cf2d07ae1e5644a665c81db3fe20af9470"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out", default="reports/class-aware-sampling")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    split_dir = Path(cfg["splits_dir"])
    train_path, val_path = split_dir / "train_image_ids.txt", split_dir / "val_image_ids.txt"
    hashes = {"train": hashlib.sha256(train_path.read_bytes()).hexdigest(),
              "val": hashlib.sha256(val_path.read_bytes()).hexdigest()}
    if hashes != {"train": TRAIN_HASH, "val": VAL_HASH}:
        raise ValueError("Frozen train/val manifests changed")
    train_ids = train_path.read_text().split()
    if len(train_ids) != 8792 or len(set(train_ids)) != 8792:
        raise ValueError("Expected 8792 unique frozen training images")
    table = read_label_table(Path(cfg["data_root"]) / cfg["train_csv"])
    # No validation or held-out test label is used to generate sampling weights.
    table = table[table.ImageId.isin(set(train_ids))]
    labels = labels_by_image(table)
    started = time.perf_counter()
    index = build_patch_index(train_ids, labels)
    stats = index_statistics(index)
    if stats["candidate_patches"] != 35168:
        raise ValueError("Expected four audited candidate tiles per training image")
    known = json.loads(Path("reports/pilot/patch_balance.json").read_text())
    if not np.isclose(stats["background_patch_ratio"], known["all_background_ratio"], atol=1e-12):
        raise ValueError("E2 train-only tile index differs from audited E0 patch balance")
    np.savez_compressed(out / "train_patch_index.npz", image_ids=np.asarray(train_ids),
                        lefts=np.asarray(index["lefts"]), flags=index["flags"],
                        normal_source=index["normal_source"])
    draws = {}
    for epoch in (1, 2):
        sample = draw_epoch(index, stats["class_patch_ratio"], seed=cfg["seed"], epoch=epoch)
        draws[str(epoch)] = draw_statistics(index, sample)
    record = {"scope": "train labels only; four fixed 256x512 candidate tiles per training original",
              "manifest_hashes": hashes, "seed": cfg["seed"], "frequency_weight": "inverse square root of class patch frequency, additive for multilabel",
              "strata": "normal-source flag and count of positive candidate tiles; stratum draw counts unchanged",
              "replacement_fraction": .25, "max_uses_per_image_per_epoch": 3,
              "random_crop_after_source_sampling": "unchanged E0 RandomCropOrResize(256,512)",
              "index_statistics": stats, "planned_full_train_draw_statistics": draws,
              "build_seconds": time.perf_counter() - started}
    (out / "sampling_design.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
