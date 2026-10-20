#!/usr/bin/env python3
"""Two-epoch E1 pilot on the previously fixed E0 pilot train/val IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steel_defect_segmentation.dataset import RandomCropOrResize, SteelDefectDataset
from steel_defect_segmentation.labels import decode_multilabel, labels_by_image, read_label_table
from steel_defect_segmentation.losses import BCEDiceLoss
from steel_defect_segmentation.metrics import ConfusionAccumulator
from steel_defect_segmentation.model import ResNet18UNet
from steel_defect_segmentation.tiling import stitch_logits, tile_image


def evaluate(model, image_ids, root, cfg, labels, device, criterion):
    model.eval()
    confusion = ConfusionAccumulator()
    bce_values, dice_values, total_values = [], [], []
    image_pixels = 0
    background_predictions = 0
    with torch.inference_mode():
        for image_id in image_ids:
            path = root / cfg["image_dir"] / image_id
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(path)
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            target = decode_multilabel(labels.get(image_id, {}), *image.shape[:2]).astype(np.float32)
            positions, logits_patches = [], []
            for top, left, patch in tile_image(image, 256, 512, 256, 384):
                input_tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))).float().div_(255).unsqueeze(0).to(device)
                logits_patches.append(model(input_tensor)[0].cpu().numpy())
                positions.append((top, left))
            logits = torch.from_numpy(stitch_logits(logits_patches, positions, *image.shape[:2])).unsqueeze(0)
            target_tensor = torch.from_numpy(target).unsqueeze(0)
            bce, dice, total = criterion.components(logits, target_tensor)
            if not all(torch.isfinite(x) for x in (bce, dice, total)):
                raise FloatingPointError(f"Nonfinite validation loss for {image_id}")
            bce_values.append(bce.item())
            dice_values.append(dice.item())
            total_values.append(total.item())
            confusion.update(logits, target_tensor, threshold=0.5)
            image_pixels += image.shape[0] * image.shape[1]
            background_predictions += int(not (torch.sigmoid(logits) >= 0.5).any().item())
    metrics = confusion.compute()
    metrics.update({
        "val_bce_loss": float(np.mean(bce_values)),
        "val_dice_loss": float(np.mean(dice_values)),
        "val_total_loss": float(np.mean(total_values)),
        "prediction_positive_pixel_ratio_per_class": [(tp + fp) / image_pixels for tp, fp in zip(metrics["tp"], metrics["fp"])],
        "full_background_prediction_image_ratio": background_predictions / len(image_ids),
    })
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out", default="reports/bce-dice-pilot")
    parser.add_argument("--e0-pilot", default="reports/pilot")
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"E1 pilot output already exists: {output}")
    cfg = yaml.safe_load(Path(args.config).read_text())
    assert cfg["loader"]["batch_size"] == 4
    assert cfg["training"]["learning_rate"] == 3e-4
    assert cfg["training"]["threshold"] == 0.5
    e0_pilot = Path(args.e0_pilot)
    train_ids = (e0_pilot / "pilot_train_ids.txt").read_text().split()
    val_ids = (e0_pilot / "pilot_val_ids.txt").read_text().split()
    split_dir = Path(cfg["splits_dir"])
    frozen_train = set((split_dir / "train_image_ids.txt").read_text().split())
    frozen_val = set((split_dir / "val_image_ids.txt").read_text().split())
    if len(train_ids) != 16 or len(val_ids) != 12 or not set(train_ids) <= frozen_train or not set(val_ids) <= frozen_val or set(train_ids) & set(val_ids):
        raise ValueError("Fixed E0 pilot IDs do not match frozen train/val manifests")
    root = Path(cfg["data_root"])
    labels = labels_by_image(read_label_table(root / cfg["train_csv"]))
    coverage = {}
    for name, ids in (("train", train_ids), ("val", val_ids)):
        coverage[name] = {"normal": sum(not labels.get(i, {}) for i in ids), "class_images": [sum(isinstance(labels.get(i, {}).get(c), str) and bool(labels[i][c].strip()) for i in ids) for c in range(1, 5)]}
        if coverage[name]["normal"] == 0 or not all(coverage[name]["class_images"]):
            raise ValueError(f"Pilot subset lacks a class or normal image: {name}")
    output.mkdir(parents=True)
    (output / "pilot_train_ids.txt").write_text("\n".join(train_ids) + "\n")
    (output / "pilot_val_ids.txt").write_text("\n".join(val_ids) + "\n")
    cfg_record = {
        "base_config": cfg,
        "experiment": "E1 BCE plus 1.0 Soft Dice",
        "loss_difference_from_E0": "BCEWithLogitsLoss + 1.0 * SoftDiceLoss",
        "dice_absent_class_rule": "Exclude classes without a positive target pixel in the current batch; all-empty batch Dice loss is zero",
        "early_stopping_monitor_for_formal_training": "full validation BCE, patience 6",
        "pilot_epochs": 2,
        "pilot_id_source": str(e0_pilot),
        "pilot_id_sha256": {"train": hashlib.sha256((e0_pilot / "pilot_train_ids.txt").read_bytes()).hexdigest(), "val": hashlib.sha256((e0_pilot / "pilot_val_ids.txt").read_bytes()).hexdigest()},
        "coverage": coverage,
    }
    (output / "config.yaml").write_text(yaml.safe_dump(cfg_record, sort_keys=False))
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dataset = SteelDefectDataset(root / cfg["image_dir"], train_ids, labels, RandomCropOrResize(256, 512, seed))
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)
    model = ResNet18UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=cfg["training"]["weight_decay"])
    criterion = BCEDiceLoss(dice_weight=1.0)
    history = []
    best_bce, best_total, best_macro = float("inf"), float("inf"), float("-inf")
    started = time.perf_counter()
    for epoch in (1, 2):
        model.train()
        train_bce, train_dice, train_total = [], [], []
        gradient_finite = True
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device, dtype=torch.float32)
            if images.shape[1:] != (3, 256, 512) or targets.shape[1:] != (4, 256, 512):
                raise ValueError("Unexpected pilot tensor shape")
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            bce, dice, total = criterion.components(logits, targets)
            if not all(torch.isfinite(x) for x in (bce, dice, total)):
                raise FloatingPointError(f"Nonfinite E1 train loss at epoch {epoch}")
            total.backward()
            gradient_finite = gradient_finite and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            if not gradient_finite:
                raise FloatingPointError(f"Nonfinite E1 gradient at epoch {epoch}")
            optimizer.step()
            train_bce.append(bce.detach().item())
            train_dice.append(dice.detach().item())
            train_total.append(total.detach().item())
        metrics = evaluate(model, val_ids, root, cfg, labels, device, criterion)
        metrics.update({
            "epoch": epoch,
            "train_bce_loss": float(np.mean(train_bce)),
            "train_dice_loss": float(np.mean(train_dice)),
            "train_total_loss": float(np.mean(train_total)),
            "gradient_finite": gradient_finite,
            "train_samples": len(train_ids), "train_batches": len(loader),
            "val_samples": len(val_ids), "val_tiles": len(val_ids) * 4,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        history.append(metrics)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": metrics}
        torch.save(state, output / "last.pt")
        if metrics["val_bce_loss"] < best_bce:
            best_bce = metrics["val_bce_loss"]
            torch.save(state, output / "best_val_bce.pt")
        if metrics["val_total_loss"] < best_total:
            best_total = metrics["val_total_loss"]
            torch.save(state, output / "best_val_total.pt")
        if metrics["macro_dice"] > best_macro:
            best_macro = metrics["macro_dice"]
            torch.save(state, output / "best_macro_dice.pt")
        (output / "metrics.json").write_text(json.dumps(history, indent=2))
        print(json.dumps({"epoch": epoch, "train_total_loss": metrics["train_total_loss"], "val_bce_loss": metrics["val_bce_loss"], "val_total_loss": metrics["val_total_loss"], "macro_dice": metrics["macro_dice"]}), flush=True)
    summary = {
        "device": str(device), "epochs": 2, "elapsed_seconds": time.perf_counter() - started,
        "mps_current_allocated_bytes": torch.mps.current_allocated_memory() if device.type == "mps" else None,
        "best_val_bce_epoch": min(history, key=lambda x: (x["val_bce_loss"], x["epoch"]))["epoch"],
        "best_val_total_epoch": min(history, key=lambda x: (x["val_total_loss"], x["epoch"]))["epoch"],
        "best_macro_dice_epoch": max(history, key=lambda x: (x["macro_dice"], -x["epoch"]))["epoch"],
        "selection_rule": "Strict improvement; ties keep earlier epoch",
        "coverage": coverage,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
