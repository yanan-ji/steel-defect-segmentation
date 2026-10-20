#!/usr/bin/env python3
"""Two-epoch E2 sampler Pilot; never launches formal training."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steel_defect_segmentation.dataset import RandomCropOrResize, SteelDefectDataset
from steel_defect_segmentation.labels import labels_by_image, read_label_table
from steel_defect_segmentation.model import ResNet18UNet
from steel_defect_segmentation.sampling import draw_epoch, draw_statistics, index_statistics
from train_e0_formal import atomic_checkpoint, log, rng_state, validate


def pilot_subset(full_index, previous_ids, size, seed):
    """Preserve source-normal ratio while ensuring minority-class coverage."""
    ids = full_index["image_ids"]
    positions = {image_id: i for i, image_id in enumerate(ids)}
    if not set(previous_ids) <= positions.keys():
        raise ValueError("Prior pilot IDs are outside frozen training manifest")
    rng = np.random.default_rng(seed)
    normal = full_index["normal_source"]
    flags = full_index["flags"].any(axis=1)
    normal_target = round(size * float(normal.mean()))
    defect_target = size - normal_target
    picked = [positions[image_id] for image_id in previous_ids]
    if len(set(picked)) != len(picked):
        raise ValueError("Duplicate prior pilot ID")
    chosen = set(picked)
    normal_picked = [i for i in picked if normal[i]]
    defect_picked = [i for i in picked if not normal[i]]

    def add_from(pool, desired, destination):
        candidates = np.asarray([i for i in pool if i not in chosen], dtype=np.int64)
        rng.shuffle(candidates)
        for i in candidates[:max(0, desired - len(destination))]:
            destination.append(int(i))
            chosen.add(int(i))

    for class_id, desired in ((1, 20), (0, 30), (3, 30)):
        existing = sum(bool(flags[i, class_id]) for i in defect_picked)
        candidates = np.flatnonzero(flags[:, class_id] & ~normal)
        candidates = np.asarray([i for i in candidates if i not in chosen], dtype=np.int64)
        rng.shuffle(candidates)
        for i in candidates[:max(0, desired - existing)]:
            defect_picked.append(int(i))
            chosen.add(int(i))
    if len(defect_picked) > defect_target:
        raise ValueError("Pilot minority-class quotas exceed defect-source budget")
    add_from(np.flatnonzero(~normal), defect_target, defect_picked)
    add_from(np.flatnonzero(normal), normal_target, normal_picked)
    selected = np.asarray(normal_picked + defect_picked, dtype=np.int64)
    if len(selected) != size or len(set(selected)) != size:
        raise RuntimeError("Pilot subset selection failed")
    rng.shuffle(selected)
    return {"image_ids": [ids[i] for i in selected], "lefts": full_index["lefts"],
            "flags": full_index["flags"][selected], "normal_source": normal[selected]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out", default="reports/class-aware-sampling/pilot")
    parser.add_argument("--index", default="reports/class-aware-sampling/train_patch_index.npz")
    parser.add_argument("--train-size", type=int, default=256)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Pilot output already exists: {output}")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg["loader"]["batch_size"] != 4 or cfg["training"]["learning_rate"] != 3e-4 or cfg["training"]["threshold"] != .5:
        raise ValueError("E2 Pilot must retain E0 batch size, LR and evaluation threshold")
    design = json.loads(Path(args.index).with_name("sampling_design.json").read_text())
    split_dir = Path(cfg["splits_dir"])
    hashes = {name: hashlib.sha256((split_dir / f"{name}_image_ids.txt").read_bytes()).hexdigest()
              for name in ("train", "val")}
    if hashes != design["manifest_hashes"]:
        raise ValueError("Sampling index and frozen train/val manifests differ")
    full_ids = (split_dir / "train_image_ids.txt").read_text().split()
    stored = np.load(args.index)
    if stored["image_ids"].tolist() != full_ids:
        raise ValueError("Stored index IDs differ from frozen training manifest")
    full_index = {"image_ids": full_ids, "lefts": stored["lefts"].tolist(),
                  "flags": stored["flags"], "normal_source": stored["normal_source"]}
    prior = Path("reports/pilot/pilot_train_ids.txt").read_text().split()
    subset = pilot_subset(full_index, prior, args.train_size, cfg["seed"])
    val_ids = Path("reports/pilot/pilot_val_ids.txt").read_text().split()
    frozen_val = set((split_dir / "val_image_ids.txt").read_text().split())
    if len(val_ids) != 12 or not set(val_ids) <= frozen_val or set(val_ids) & set(subset["image_ids"]):
        raise ValueError("Prior pilot validation IDs are not frozen validation members")
    root = Path(cfg["data_root"])
    needed = set(subset["image_ids"]) | set(val_ids)
    table = read_label_table(root / cfg["train_csv"])
    labels = labels_by_image(table[table.ImageId.isin(needed)])
    output.mkdir(parents=True)
    (output / "pilot_train_ids.txt").write_text("\n".join(subset["image_ids"]) + "\n")
    (output / "pilot_val_ids.txt").write_text("\n".join(val_ids) + "\n")
    pilot_index_stats = index_statistics(subset)
    record = {"base_e0_config": cfg, "experiment": "E2 class-aware source-image sampling only",
              "unchanged": "ResNet18 U-Net scratch seed 42, BCEWithLogitsLoss, AdamW lr 3e-4, batch 4, random 256x512 crop, full-image validation, 0.50 evaluation",
              "formal_early_stopping": "full validation BCE, patience 6, max 30 epochs",
              "pilot_epochs": 2, "pilot_train_size": len(subset["image_ids"]), "pilot_val_size": len(val_ids),
              "pilot_subset": "prior E0 Pilot IDs plus stratified frozen-train IDs; not used for formal model selection",
              "manifest_hashes": hashes, "sampling_design": design,
              "pilot_candidate_index_statistics": pilot_index_stats,
              "pilot_id_hashes": {"train": hashlib.sha256((output / "pilot_train_ids.txt").read_bytes()).hexdigest(),
                                  "val": hashlib.sha256((output / "pilot_val_ids.txt").read_bytes()).hexdigest()}}
    (output / "config.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type != "mps":
        raise RuntimeError("E2 Pilot requires MPS on this platform")
    model = ResNet18UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
    criterion = nn.BCEWithLogitsLoss()
    crop = RandomCropOrResize(256, 512, seed)
    loader_generator = torch.Generator().manual_seed(seed)
    class_frequency = design["index_statistics"]["class_patch_ratio"]
    history = []
    best_bce, best_macro = float("inf"), float("-inf")
    total_start = time.perf_counter()
    for epoch in (1, 2):
        indices = draw_epoch(subset, class_frequency, seed, epoch)
        planned = draw_statistics(subset, indices)
        sampled_ids = [subset["image_ids"][i] for i in indices]
        dataset = SteelDefectDataset(root / cfg["image_dir"], sampled_ids, labels, crop)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
                            pin_memory=False, generator=loader_generator)
        model.train()
        started = time.perf_counter()
        loss_total, samples = 0.0, 0
        class_counts = np.zeros(4, np.int64)
        background_count = 0
        gradient_finite = True
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device, dtype=torch.float32)
            if images.shape[1:] != (3, 256, 512) or targets.shape[1:] != (4, 256, 512):
                raise ValueError("Unexpected E2 Pilot tensor shapes")
            patch_flags = targets.gt(0).any(dim=(2, 3)).cpu().numpy()
            class_counts += patch_flags.sum(axis=0)
            background_count += int((~patch_flags.any(axis=1)).sum())
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite E2 Pilot loss epoch {epoch}")
            loss.backward()
            gradient_finite &= all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            if not gradient_finite:
                raise FloatingPointError(f"Nonfinite E2 Pilot gradient epoch {epoch}")
            optimizer.step()
            loss_total += loss.detach().item() * len(images)
            samples += len(images)
        train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        metrics = validate(model, val_ids, root, cfg, labels, device, output, epoch, objective="bce")
        val_seconds = time.perf_counter() - started
        improved_bce = metrics["val_loss"] < best_bce
        improved_macro = metrics["macro_dice"] > best_macro
        best_bce = min(best_bce, metrics["val_loss"])
        best_macro = max(best_macro, metrics["macro_dice"])
        scheduler.step()
        metrics.update({"epoch": epoch, "train_bce_loss": loss_total / samples,
                        "train_samples": samples, "train_batches": len(loader),
                        "val_samples": len(val_ids), "train_seconds": train_seconds,
                        "validation_seconds": val_seconds, "gradient_finite": gradient_finite,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "sampling": {"candidate_draw": planned,
                                     "actual_random_crop_background_ratio": background_count / samples,
                                     "actual_random_crop_class_ratio": (class_counts / samples).tolist(),
                                     "actual_random_crop_class_counts": class_counts.tolist()},
                        "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
                        "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory()})
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": epoch,
                 "metrics": metrics, "manifest_hashes": hashes,
                 "pilot_id_hashes": record["pilot_id_hashes"], "objective": "bce",
                 "sampler": "class-aware", "rng": rng_state(crop, loader_generator)}
        atomic_checkpoint(output / "last.pt", state)
        if improved_bce:
            atomic_checkpoint(output / "best_val_loss.pt", state)
        if improved_macro:
            atomic_checkpoint(output / "best_macro_dice.pt", state)
        history.append(metrics)
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics, allow_nan=False) + "\n")
            f.flush()
        log(f"E2 Pilot epoch {epoch}: train BCE={metrics['train_bce_loss']:.6f}, val BCE={metrics['val_loss']:.6f}, macro Dice={metrics['macro_dice']:.6f}, actual background={background_count/samples:.3f}", output / "run.log")
    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    ResNet18UNet().load_state_dict(checkpoint["model"], strict=True)
    if checkpoint["epoch"] != 2 or checkpoint["manifest_hashes"] != hashes:
        raise ValueError("E2 Pilot checkpoint round-trip failed")
    summary = {"status": "pilot_completed", "epochs": 2, "device": str(device),
               "elapsed_seconds": time.perf_counter() - total_start,
               "train_subset_images": len(subset["image_ids"]), "val_subset_images": len(val_ids),
               "best_val_bce_epoch": min(history, key=lambda x: (x["val_loss"], x["epoch"]))["epoch"],
               "best_macro_dice_epoch": max(history, key=lambda x: (x["macro_dice"], -x["epoch"]))["epoch"],
               "checkpoint_round_trip": True,
               "pilot_is_not_formal_performance_comparison": True}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
