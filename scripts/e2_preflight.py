#!/usr/bin/env python3
"""E2 preflight and deterministic E0 crop-distribution replay; no model training."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steel_defect_segmentation.dataset import RandomCropOrResize, SteelDefectDataset
from steel_defect_segmentation.labels import labels_by_image, read_label_table
from steel_defect_segmentation.model import ResNet18UNet
from steel_defect_segmentation.sampling import draw_epoch, draw_statistics, index_statistics


EXPECTED = {"train": "ba7887219c3d07215f44b704f053dff6bfece91375bd7a5178353af156152c89",
            "val": "593cf50fe4c05f2d788f3fb8a15a55cf2d07ae1e5644a665c81db3fe20af9470",
            "test": "79ebfdf0244af6bf001f84de3443166fa84f2d05591a76fdea20684a66120826"}


def replay_e0_crop_stats(cfg, train_ids, labels):
    crop = RandomCropOrResize(256, 512, cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    dataset = SteelDefectDataset(Path(cfg["data_root"]) / cfg["image_dir"], train_ids, labels, crop)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False, generator=generator)
    class_counts = np.zeros(4, np.int64)
    background_count = 0
    total = 0
    for batch in loader:
        flags = batch["mask"].gt(0).any(dim=(2, 3)).numpy()
        class_counts += flags.sum(axis=0)
        background_count += int((~flags.any(axis=1)).sum())
        total += len(flags)
    return {"estimate_type": "deterministic offline replay of E0 seed-42 DataLoader shuffle and RandomCropOrResize; not recorded in original E0 logs",
            "samples": total, "batches": len(loader), "background_patch_count": background_count,
            "background_patch_ratio": background_count / total,
            "class_positive_patch_counts": class_counts.tolist(),
            "class_positive_patch_ratios": (class_counts / total).tolist(),
            "any_defect_patch_ratio": 1 - background_count / total,
            "normal_source_image_ratio": float(np.mean([not labels.get(image_id, {}) for image_id in train_ids]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--index", default="reports/class-aware-sampling/train_patch_index.npz")
    parser.add_argument("--out", default="reports/class-aware-sampling-formal/preflight.json")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    e0 = yaml.safe_load(Path("reports/baseline-bce-formal/config.yaml").read_text())
    split_dir = Path(cfg["splits_dir"])
    hashes = {name: hashlib.sha256((split_dir / f"{name}_image_ids.txt").read_bytes()).hexdigest()
              for name in ("train", "val", "test")}
    if hashes != EXPECTED:
        raise ValueError("A frozen manifest hash changed")
    train_ids = (split_dir / "train_image_ids.txt").read_text().split()
    val_ids = (split_dir / "val_image_ids.txt").read_text().split()
    if len(train_ids) != 8792 or len(val_ids) != 1881 or len(set(train_ids)) != len(train_ids) or set(train_ids) & set(val_ids):
        raise ValueError("Unexpected frozen train/val manifests")
    for key in ("data_root", "train_csv", "image_dir", "splits_dir", "seed", "loader", "patch", "training"):
        if cfg[key] != e0[key]:
            raise ValueError(f"E2/E0 baseline config mismatch: {key}")
    if (cfg["seed"], cfg["loader"]["batch_size"], cfg["training"]["learning_rate"],
        cfg["training"]["epochs"], cfg["training"]["patience"], cfg["training"]["threshold"]) != (42, 4, 3e-4, 30, 6, .5):
        raise ValueError("E2 formal controls differ from E0")
    stored = np.load(args.index)
    if stored["image_ids"].tolist() != train_ids or stored["flags"].shape != (8792, 4, 4):
        raise ValueError("E2 patch index does not align to full frozen train IDs")
    index = {"image_ids": train_ids, "lefts": stored["lefts"].tolist(),
             "flags": stored["flags"], "normal_source": stored["normal_source"]}
    design = json.loads(Path(args.index).with_name("sampling_design.json").read_text())
    if design["manifest_hashes"] != {"train": hashes["train"], "val": hashes["val"]}:
        raise ValueError("E2 train-only sampling weights use stale split hashes")
    table = read_label_table(Path(cfg["data_root"]) / cfg["train_csv"])
    labels = labels_by_image(table[table.ImageId.isin(set(train_ids))])
    e0_stats = replay_e0_crop_stats(cfg, train_ids, labels)
    planned = {}
    for epoch in (1, 2):
        draw = draw_epoch(index, design["index_statistics"]["class_patch_ratio"], cfg["seed"], epoch)
        planned[str(epoch)] = draw_statistics(index, draw)
    result = {"manifest_sha256": hashes,
              "test_manifest_access": "SHA256 only; test IDs were not parsed and no test images or labels were accessed",
              "training_control_comparison": {"model": "same ResNet18UNet(classes=4), 14,274,244 parameters, seed 42 scratch initialization",
                  "loss": "same BCEWithLogitsLoss; no Dice or pos_weight",
                  "optimizer": "same AdamW, lr=3e-4, weight_decay from E0 config",
                  "batch_size": 4, "patch": "same synchronized random crop 256x512; masks nearest only",
                  "max_epochs": 30, "early_stopping": "full validation BCE, patience 6",
                  "threshold": .5, "sole_intended_change": "training source-image sampling strategy"},
              "samples_per_epoch": {"E0": 8792, "E2": int(len(train_ids)),
                                    "E2_batches": int(np.ceil(len(train_ids) / 4))},
              "E0_actual_crop_distribution_estimate": e0_stats,
              "E0_candidate_tile_distribution": json.loads(Path("reports/pilot/patch_balance.json").read_text()),
              "E2_candidate_tile_distribution": index_statistics(index),
              "E2_planned_source_sampling": planned,
              "note": "Candidate-tile frequencies predict source sampling only; the actual random-crop mask distribution is measured during E2 training."}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
