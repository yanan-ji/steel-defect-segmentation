#!/usr/bin/env python3
"""Full-data E0/E1 experiments with shared data and training controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steel_defect_segmentation.dataset import RandomCropOrResize, SteelDefectDataset
from steel_defect_segmentation.labels import decode_multilabel, labels_by_image, read_label_table
from steel_defect_segmentation.losses import BCEDiceLoss
from steel_defect_segmentation.metrics import ConfusionAccumulator
from steel_defect_segmentation.model import ResNet18UNet
from steel_defect_segmentation.tiling import stitch_logits, tile_image


def log(message: str, path: Path) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_checkpoint(path: Path, state: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def rng_state(crop: RandomCropOrResize, loader_generator: torch.Generator) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "crop": crop.rng.bit_generator.state,
        "loader": loader_generator.get_state(),
    }
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state: dict, crop: RandomCropOrResize, loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    crop.rng.bit_generator.state = state["crop"]
    loader_generator.set_state(state["loader"])
    if "torch_mps" in state:
        torch.mps.set_rng_state(state["torch_mps"])


def draw_case(path: Path, image: np.ndarray, target: np.ndarray, probabilities: np.ndarray, class_id: int) -> None:
    color = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    truth, prediction = color.copy(), color.copy()
    truth[target[class_id]] = (0, 255, 0)
    prediction[probabilities[class_id] >= .5] = (0, 0, 255)
    heat = cv2.applyColorMap(np.uint8(np.clip(probabilities[class_id], 0, 1) * 255), cv2.COLORMAP_JET)
    cv2.imwrite(str(path), cv2.hconcat((truth, prediction, heat)))


def validate(model, image_ids, root, cfg, labels, device, output_dir, epoch, objective="bce"):
    model.eval()
    accumulator = ConfusionAccumulator()
    criterion = nn.BCEWithLogitsLoss()
    loss_sum = 0.0
    dice_sum = 0.0
    dice_criterion = BCEDiceLoss(1.0).dice if objective == "bce-dice" else None
    pixel_total = 0
    probability_sums = np.zeros(4, dtype=np.float64)
    histogram = np.zeros((4, 20), dtype=np.int64)
    background_images = 0
    examples = {c: {"fn": None, "fp": None, "tp": None} for c in range(4)}
    example_dir = output_dir / "visualizations" / f"epoch_{epoch:02d}"
    example_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, image_id in enumerate(image_ids, 1):
            path = root / cfg["image_dir"] / image_id
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(path)
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            if (height, width) != (256, 1600):
                raise ValueError(f"Unexpected image size {image_id}: {(height, width)}")
            target = decode_multilabel(labels.get(image_id, {}), height, width).astype(np.float32)
            positions, parts = [], []
            for top, left, patch in tile_image(image, 256, 512, 256, 384):
                tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))).float().div_(255).unsqueeze(0).to(device)
                parts.append(model(tensor)[0].cpu().numpy())
                positions.append((top, left))
            logits = torch.from_numpy(stitch_logits(parts, positions, height, width)).unsqueeze(0)
            truth = torch.from_numpy(target).unsqueeze(0)
            loss = criterion(logits, truth)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite validation loss: {image_id}")
            loss_sum += float(loss)
            if dice_criterion is not None:
                dice_value = dice_criterion(logits, truth)
                if not torch.isfinite(dice_value):
                    raise FloatingPointError(f"Nonfinite validation Dice loss: {image_id}")
                dice_sum += float(dice_value)
            accumulator.update(logits, truth, .5)
            probabilities = torch.sigmoid(logits)[0].numpy()
            prediction = probabilities >= .5
            pixel_total += height * width
            background_images += int(not prediction.any())
            probability_sums += probabilities.sum(axis=(1, 2), dtype=np.float64)
            for class_id in range(4):
                histogram[class_id] += np.histogram(probabilities[class_id], bins=20, range=(0, 1))[0]
                gt = target[class_id].astype(bool)
                if examples[class_id]["fn"] is None and np.any(gt & ~prediction[class_id]):
                    examples[class_id]["fn"] = image_id
                    draw_case(example_dir / f"class_{class_id+1}_fn_{image_id}", image, gt[None].repeat(4, axis=0), probabilities, class_id)
                if examples[class_id]["fp"] is None and np.any(~gt & prediction[class_id]):
                    examples[class_id]["fp"] = image_id
                    draw_case(example_dir / f"class_{class_id+1}_fp_{image_id}", image, gt[None].repeat(4, axis=0), probabilities, class_id)
                if examples[class_id]["tp"] is None and np.any(gt & prediction[class_id]):
                    examples[class_id]["tp"] = image_id
                    draw_case(example_dir / f"class_{class_id+1}_tp_{image_id}", image, gt[None].repeat(4, axis=0), probabilities, class_id)
            if index % 200 == 0:
                log(f"epoch {epoch} validation images {index}/{len(image_ids)}", output_dir / "run.log")
    metrics = accumulator.compute()
    metrics.update({
        "val_loss": loss_sum / len(image_ids),
        "prediction_positive_pixel_ratio_per_class": [(tp + fp) / pixel_total for tp, fp in zip(metrics["tp"], metrics["fp"])],
        "probability_mean_per_class": (probability_sums / pixel_total).tolist(),
        "probability_histogram_20_bins_per_class": histogram.tolist(),
        "full_background_prediction_image_ratio": background_images / len(image_ids),
        "representative_examples": examples,
    })
    if dice_criterion is not None:
        metrics["val_bce_loss"] = metrics["val_loss"]
        metrics["val_dice_loss"] = dice_sum / len(image_ids)
        metrics["val_total_loss"] = (loss_sum + dice_sum) / len(image_ids)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--objective", choices=("bce", "bce-dice"), default="bce")
    parser.add_argument("--out")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    assert cfg["loader"]["batch_size"] == 4
    assert cfg["training"]["learning_rate"] == 3e-4
    assert cfg["training"]["threshold"] == .5
    assert cfg["training"]["epochs"] == 30
    output_dir = Path(args.out or ("reports/bce-dice-formal" if args.objective == "bce-dice" else "reports/baseline-bce-formal"))
    if args.objective == "bce-dice" and output_dir.resolve() == Path("reports/baseline-bce-formal").resolve():
        raise ValueError("E1 output must not overwrite the E0 experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(cfg["splits_dir"])
    train_path, val_path = manifest_dir / "train_image_ids.txt", manifest_dir / "val_image_ids.txt"
    train_ids, val_ids = train_path.read_text().split(), val_path.read_text().split()
    if len(train_ids) != 8792 or len(val_ids) != 1881 or set(train_ids) & set(val_ids):
        raise ValueError("Frozen train/val manifests differ from audited E0 manifests")
    manifest_hash = {"train": hashlib.sha256(train_path.read_bytes()).hexdigest(), "val": hashlib.sha256(val_path.read_bytes()).hexdigest()}
    if manifest_hash != {"train": "ba7887219c3d07215f44b704f053dff6bfece91375bd7a5178353af156152c89", "val": "593cf50fe4c05f2d788f3fb8a15a55cf2d07ae1e5644a665c81db3fe20af9470"}:
        raise ValueError("Frozen manifest hashes changed")
    root = Path(cfg["data_root"])
    labels = labels_by_image(read_label_table(root / cfg["train_csv"]))
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    crop = RandomCropOrResize(cfg["patch"]["height"], cfg["patch"]["width"], seed)
    loader_generator = torch.Generator().manual_seed(seed)
    dataset = SteelDefectDataset(root / cfg["image_dir"], train_ids, labels, crop)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False, generator=loader_generator)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if args.objective == "bce-dice" and device.type != "mps":
        raise RuntimeError("E1 formal experiment requires an available MPS device")
    model = ResNet18UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=cfg["training"]["weight_decay"])
    # Explicit constant-rate scheduler keeps the E0 learning rate fixed and has restorable state.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
    criterion = nn.BCEWithLogitsLoss() if args.objective == "bce" else BCEDiceLoss(1.0)
    last_path = output_dir / "last.pt"
    history_path = output_dir / "metrics.jsonl"
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(last_path)
        # RNG snapshots must remain CPU ByteTensors; model/optimizer loaders
        # transfer parameter state to the active device as appropriate.
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("objective", "bce") != args.objective:
            raise ValueError("Resume objective mismatch")
        if state["manifest_hash"] != manifest_hash:
            raise ValueError("Resume manifest hash mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"], crop, loader_generator)
        epoch_start = state["epoch"] + 1
        best_loss, best_dice = state["best_val_loss"], state["best_macro_dice"]
        best_total = state.get("best_val_total", math.inf)
        stale = state["early_stopping_stale"]
        history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
        history = [row for row in history if row["epoch"] <= state["epoch"]]
        history_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history))
    else:
        if last_path.exists() or history_path.exists():
            raise FileExistsError("Independent experiment output already contains a run; use --resume")
        epoch_start, best_loss, best_dice, best_total, stale, history = 1, math.inf, -math.inf, math.inf, 0, []
        config_record = dict(cfg)
        config_record["objective"] = args.objective
        if args.objective == "bce-dice":
            config_record["dice_weight"] = 1.0
            config_record["dice_absent_class_rule"] = "exclude classes without positive target pixels; all-empty batch Dice is zero"
            config_record["early_stopping_monitor"] = "full-image validation BCE"
        (output_dir / "config.yaml").write_text(yaml.safe_dump(config_record, sort_keys=False))
        (output_dir / "manifest_hashes.json").write_text(json.dumps(manifest_hash, indent=2))
    log(f"starting {args.objective} at epoch {epoch_start}, device={device}, train={len(train_ids)}, val={len(val_ids)}", output_dir / "run.log")
    for epoch in range(epoch_start, 31):
        if stale >= 6:
            log(f"early stopping already reached at epoch {epoch-1}", output_dir / "run.log")
            break
        started = time.perf_counter()
        model.train()
        loss_sum, sample_count = 0.0, 0
        bce_sum, dice_sum = 0.0, 0.0
        gradient_ok = True
        dice_gradient_checked = False
        if device.type == "mps" and hasattr(torch.mps, "reset_peak_memory_stats"):
            torch.mps.reset_peak_memory_stats()
        for batch_index, batch in enumerate(loader, 1):
            images = batch["image"].to(device)
            targets = batch["mask"].to(device, dtype=torch.float32)
            if images.shape[1:] != (3, 256, 512) or targets.shape[1:] != (4, 256, 512):
                raise ValueError("Unexpected training tensor shapes")
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if args.objective == "bce-dice":
                bce_loss, dice_loss, loss = criterion.components(logits, targets)
                if not all(bool(torch.isfinite(value)) for value in (bce_loss, dice_loss, loss)):
                    raise FloatingPointError(f"Nonfinite E1 loss at epoch {epoch}, batch {batch_index}")
                if epoch == 1 and not dice_gradient_checked and bool(targets.any()):
                    dice_gradient = torch.autograd.grad(dice_loss, logits, retain_graph=True)[0]
                    if not bool(torch.isfinite(dice_gradient).all()) or not bool((dice_gradient.abs().sum() > 0)):
                        raise FloatingPointError(f"Invalid Dice gradient at epoch {epoch}, batch {batch_index}")
                    dice_gradient_checked = True
                bce_sum += bce_loss.detach().item() * len(images)
                dice_sum += dice_loss.detach().item() * len(images)
            else:
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite train loss at epoch {epoch}, batch {batch_index}")
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                gradient_ok = False
                raise FloatingPointError(f"Nonfinite gradient at epoch {epoch}, batch {batch_index}")
            optimizer.step()
            loss_sum += loss.detach().item() * len(images)
            sample_count += len(images)
            if batch_index % 200 == 0:
                log(f"epoch {epoch} train batches {batch_index}/{len(loader)}", output_dir / "run.log")
        train_seconds = time.perf_counter() - started
        if args.objective == "bce-dice" and epoch == 1 and not dice_gradient_checked:
            raise RuntimeError("No positive training batch found for the E1 Dice gradient check")
        validation_started = time.perf_counter()
        metrics = validate(model, val_ids, root, cfg, labels, device, output_dir, epoch, args.objective)
        validation_seconds = time.perf_counter() - validation_started
        current_loss, current_dice = metrics["val_loss"], metrics["macro_dice"]
        improved_loss = current_loss < best_loss
        improved_dice = current_dice > best_dice
        improved_total = args.objective == "bce-dice" and metrics["val_total_loss"] < best_total
        best_loss = min(best_loss, current_loss)
        best_dice = max(best_dice, current_dice)
        if args.objective == "bce-dice":
            best_total = min(best_total, metrics["val_total_loss"])
        stale = 0 if improved_loss else stale + 1
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        metrics.update({
            "epoch": epoch, "train_loss": loss_sum / sample_count,
            "train_samples": sample_count, "train_patches": sample_count,
            "train_batches": len(loader), "val_samples": len(val_ids),
            "val_patches": len(val_ids) * 4, "learning_rate": current_lr,
            "train_seconds": train_seconds, "validation_seconds": validation_seconds,
            "epoch_seconds": time.perf_counter() - started, "gradient_finite": gradient_ok,
            "mps_current_allocated_bytes": torch.mps.current_allocated_memory() if device.type == "mps" else None,
            "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory() if device.type == "mps" else None,
            "early_stopping_stale": stale,
            "best_val_loss_epoch": epoch if improved_loss else next(row["best_val_loss_epoch"] for row in reversed(history)),
            "best_macro_dice_epoch": epoch if improved_dice else next(row["best_macro_dice_epoch"] for row in reversed(history)),
        })
        if args.objective == "bce-dice":
            metrics.update({
                "train_bce_loss": bce_sum / sample_count,
                "train_dice_loss": dice_sum / sample_count,
                "train_total_loss": loss_sum / sample_count,
                "first_epoch_dice_gradient_checked": dice_gradient_checked if epoch == 1 else None,
                "best_val_total_epoch": epoch if improved_total else next(row["best_val_total_epoch"] for row in reversed(history)),
            })
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch,
            "best_val_loss": best_loss, "best_macro_dice": best_dice,
            "best_val_total": best_total, "objective": args.objective,
            "early_stopping_stale": stale, "rng": rng_state(crop, loader_generator),
            "manifest_hash": manifest_hash, "metrics": metrics,
        }
        atomic_checkpoint(last_path, state)
        if improved_loss:
            atomic_checkpoint(output_dir / "best_val_loss.pt", state)
        if improved_dice:
            atomic_checkpoint(output_dir / "best_macro_dice.pt", state)
        if improved_total:
            atomic_checkpoint(output_dir / "best_val_total.pt", state)
        history.append(metrics)
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        log(f"epoch {epoch} complete: train_loss={metrics['train_loss']:.6f}, val_bce={current_loss:.6f}, macro_dice={current_dice:.6f}, stale={stale}, seconds={metrics['epoch_seconds']:.1f}", output_dir / "run.log")
        if stale >= 6:
            log(f"early stopping after epoch {epoch}", output_dir / "run.log")
            break
    (output_dir / "status.json").write_text(json.dumps({
        "status": "completed", "epochs_completed": history[-1]["epoch"],
        "objective": args.objective,
        "best_val_loss_epoch": history[-1]["best_val_loss_epoch"],
        "best_macro_dice_epoch": history[-1]["best_macro_dice_epoch"],
        "best_val_total_epoch": history[-1].get("best_val_total_epoch"),
        "tie_break": "strict improvement only; an exact tie keeps the earlier epoch",
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "total_epoch_seconds": sum(row["epoch_seconds"] for row in history),
    }, indent=2))


if __name__ == "__main__":
    main()
