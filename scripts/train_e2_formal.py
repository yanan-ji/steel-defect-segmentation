#!/usr/bin/env python3
"""Explicit-only full E2 trainer: E0 BCE controls with class-aware sampling."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from steel_defect_segmentation.sampling import draw_epoch, draw_statistics
from train_e0_formal import atomic_checkpoint, log, restore_rng, rng_state, validate


EXPECTED_HASHES = {"train": "ba7887219c3d07215f44b704f053dff6bfece91375bd7a5178353af156152c89",
                   "val": "593cf50fe4c05f2d788f3fb8a15a55cf2d07ae1e5644a665c81db3fe20af9470",
                   "test": "79ebfdf0244af6bf001f84de3443166fa84f2d05591a76fdea20684a66120826"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--index", default="reports/class-aware-sampling/train_patch_index.npz")
    parser.add_argument("--out", default="reports/class-aware-sampling/formal")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--checkpoint-every", type=int, default=400)
    args = parser.parse_args()
    output = Path(args.out)
    if output.resolve() in {Path("reports/baseline-bce-formal").resolve(), Path("reports/bce-dice-formal").resolve()}:
        raise ValueError("E2 must not overwrite E0/E1")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if (cfg["seed"], cfg["loader"]["batch_size"], cfg["training"]["learning_rate"],
        cfg["training"]["epochs"], cfg["training"]["patience"], cfg["training"]["threshold"]) != (42, 4, 3e-4, 30, 6, .5):
        raise ValueError("E2 formal controls differ from E0")
    split_dir = Path(cfg["splits_dir"])
    hashes = {key: hashlib.sha256((split_dir / f"{key}_image_ids.txt").read_bytes()).hexdigest()
              for key in ("train", "val", "test")}
    if hashes != EXPECTED_HASHES:
        raise ValueError("Frozen manifest hashes changed")
    train_ids = (split_dir / "train_image_ids.txt").read_text().split()
    val_ids = (split_dir / "val_image_ids.txt").read_text().split()
    if len(train_ids) != 8792 or len(val_ids) != 1881 or set(train_ids) & set(val_ids):
        raise ValueError("Unexpected frozen train/val membership")
    index_path = Path(args.index)
    index_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
    stored = np.load(index_path)
    if stored["image_ids"].tolist() != train_ids:
        raise ValueError("Train-only patch index differs from frozen train manifest")
    index = {"image_ids": train_ids, "lefts": stored["lefts"].tolist(),
             "flags": stored["flags"], "normal_source": stored["normal_source"]}
    design = json.loads(index_path.with_name("sampling_design.json").read_text())
    if design["manifest_hashes"] != {"train": hashes["train"], "val": hashes["val"]}:
        raise ValueError("Train-only index design hashes differ")
    class_frequency = design["index_statistics"]["class_patch_ratio"]
    root = Path(cfg["data_root"])
    needed = set(train_ids) | set(val_ids)
    table = read_label_table(root / cfg["train_csv"])
    labels = labels_by_image(table[table.ImageId.isin(needed)])
    if not torch.backends.mps.is_available():
        raise RuntimeError("E2 formal experiment requires MPS")
    device = torch.device("mps")
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    crop = RandomCropOrResize(256, 512, seed)
    loader_generator = torch.Generator().manual_seed(seed)
    model = ResNet18UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
    criterion = nn.BCEWithLogitsLoss()
    output.mkdir(parents=True, exist_ok=True)
    last_path, history_path = output / "last.pt", output / "metrics.jsonl"
    if args.resume:
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["objective"] != "bce" or state["sampler"] != "class-aware" or state["manifest_hashes"] != hashes or state["index_sha256"] != index_hash:
            raise ValueError("E2 resume identity mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"], crop, loader_generator)
        start_epoch = state["epoch"] if state.get("in_epoch", False) else state["epoch"] + 1
        best_bce, best_macro, stale = state["best_val_loss"], state["best_macro_dice"], state["early_stopping_stale"]
        history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
        history = [row for row in history if row["epoch"] <= state["epoch"]]
        history_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history))
    else:
        if last_path.exists() or history_path.exists():
            raise FileExistsError("Formal E2 output exists; use --resume only from complete last.pt")
        (output / "config.yaml").write_text(yaml.safe_dump({"base_e0_config": cfg,
            "only_training_difference": "class-aware source-image sampling informed by train-only patch labels",
            "sampling_design": design, "index_sha256": index_hash,
            "fixed_validation_threshold": .5, "early_stopping_monitor": "full-image validation BCE"}, sort_keys=False))
        start_epoch, best_bce, best_macro, stale, history = 1, math.inf, -math.inf, 0, []
    log(f"E2 formal start epoch={start_epoch}, train={len(train_ids)}, val={len(val_ids)}, device=mps", output / "run.log")
    for epoch in range(start_epoch, min(args.max_epochs, 30) + 1):
        if stale >= 6:
            break
        selected = draw_epoch(index, class_frequency, seed, epoch)
        planned = draw_statistics(index, selected)
        sampled_ids = [train_ids[i] for i in selected]
        resume_state = state if args.resume and epoch == start_epoch and state.get("in_epoch", False) else None
        if resume_state is not None:
            if not np.array_equal(np.asarray(resume_state["sample_plan"]), selected):
                raise ValueError("Saved E2 sampling plan does not reproduce")
            restore_rng(resume_state["rng"], crop, loader_generator)
            cursor = int(resume_state["batch_cursor"])
            loss_sum = float(resume_state["loss_sum"])
            class_counts = np.asarray(resume_state["class_counts"], dtype=np.int64)
            background_count = int(resume_state["background_count"])
            samples = int(resume_state["samples"])
            elapsed_before = float(resume_state.get("train_elapsed_seconds", 0.0))
        else:
            cursor, loss_sum, class_counts, background_count, samples, elapsed_before = 0, 0.0, np.zeros(4, np.int64), 0, 0, 0.0
        dataset = SteelDefectDataset(root / cfg["image_dir"], sampled_ids[cursor * 4:], labels, crop)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
                            pin_memory=False, generator=loader_generator)
        loader_iter = iter(loader)
        # A new iterator consumes a torch base-seed even with num_workers=0.
        # Restore after iterator creation so resumed global RNG matches the
        # uninterrupted process exactly; crop draws then continue at cursor.
        if resume_state is not None:
            restore_rng(resume_state["rng"], crop, loader_generator)
        epoch_batches = len(selected) // 4
        model.train()
        started = time.perf_counter()
        if hasattr(torch.mps, "reset_peak_memory_stats"):
            torch.mps.reset_peak_memory_stats()
        for local_batch_index, batch in enumerate(loader_iter, 1):
            batch_index = cursor + local_batch_index
            images = batch["image"].to(device)
            targets = batch["mask"].to(device, dtype=torch.float32)
            if images.shape[1:] != (3, 256, 512) or targets.shape[1:] != (4, 256, 512):
                raise ValueError("Unexpected E2 formal tensor shapes")
            patch_flags = targets.gt(0).any(dim=(2, 3)).cpu().numpy()
            class_counts += patch_flags.sum(axis=0)
            background_count += int((~patch_flags.any(axis=1)).sum())
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite E2 train loss epoch {epoch} batch {batch_index}")
            loss.backward()
            if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError(f"Nonfinite E2 gradient epoch {epoch} batch {batch_index}")
            optimizer.step()
            loss_sum += loss.detach().item() * len(images)
            samples += len(images)
            if batch_index % 200 == 0:
                log(f"epoch {epoch} train batches {batch_index}/{epoch_batches}", output / "run.log")
            if batch_index % args.checkpoint_every == 0:
                elapsed = elapsed_before + time.perf_counter() - started
                mid_state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "in_epoch": True,
                    "batch_cursor": batch_index, "sample_plan": selected,
                    "loss_sum": loss_sum, "class_counts": class_counts.tolist(),
                    "background_count": background_count, "samples": samples,
                    "train_elapsed_seconds": elapsed, "best_val_loss": best_bce,
                    "best_macro_dice": best_macro, "early_stopping_stale": stale,
                    "rng": rng_state(crop, loader_generator), "manifest_hashes": hashes,
                    "index_sha256": index_hash, "objective": "bce", "sampler": "class-aware"}
                atomic_checkpoint(last_path, mid_state)
                log(f"epoch {epoch} resumable checkpoint at batch {batch_index}/{epoch_batches}", output / "run.log")
        train_seconds = elapsed_before + time.perf_counter() - started
        validation_started = time.perf_counter()
        metrics = validate(model, val_ids, root, cfg, labels, device, output, epoch, objective="bce")
        validation_seconds = time.perf_counter() - validation_started
        improved_bce = metrics["val_loss"] < best_bce
        improved_macro = metrics["macro_dice"] > best_macro
        best_bce = min(best_bce, metrics["val_loss"])
        best_macro = max(best_macro, metrics["macro_dice"])
        stale = 0 if improved_bce else stale + 1
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        metrics.update({"epoch": epoch, "train_loss": loss_sum / samples,
                        "train_samples": samples, "train_patches": samples,
                        "train_batches": epoch_batches, "val_samples": len(val_ids),
                        "val_patches": len(val_ids) * 4, "learning_rate": lr,
                        "train_seconds": train_seconds, "validation_seconds": validation_seconds,
                        "epoch_seconds": time.perf_counter() - started,
                        "gradient_finite": True, "early_stopping_stale": stale,
                        "best_val_loss_epoch": epoch if improved_bce else history[-1]["best_val_loss_epoch"],
                        "best_macro_dice_epoch": epoch if improved_macro else history[-1]["best_macro_dice_epoch"],
                        "sampling": {"candidate_draw": planned,
                                     "actual_random_crop_background_ratio": background_count / samples,
                                     "actual_random_crop_class_ratio": (class_counts / samples).tolist(),
                                     "actual_random_crop_class_counts": class_counts.tolist()},
                        "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
                        "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory()})
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": epoch,
                 "best_val_loss": best_bce, "best_macro_dice": best_macro,
                 "early_stopping_stale": stale, "rng": rng_state(crop, loader_generator),
                 "manifest_hashes": hashes, "index_sha256": index_hash,
                 "objective": "bce", "sampler": "class-aware", "metrics": metrics,
                 "in_epoch": False, "batch_cursor": len(selected)//4,
                 "sample_plan": selected, "completed_epoch": True}
        atomic_checkpoint(last_path, state)
        if improved_bce:
            atomic_checkpoint(output / "best_val_loss.pt", state)
        if improved_macro:
            atomic_checkpoint(output / "best_macro_dice.pt", state)
        history.append(metrics)
        with history_path.open("a") as f:
            f.write(json.dumps(metrics, allow_nan=False) + "\n")
            f.flush()
        log(f"epoch {epoch} complete: train_bce={metrics['train_loss']:.6f}, val_bce={metrics['val_loss']:.6f}, macro_dice={metrics['macro_dice']:.6f}, stale={stale}", output / "run.log")
        if stale >= 6:
            log(f"early stopping after epoch {epoch}", output / "run.log")
            break
        if epoch >= args.max_epochs:
            break
    (output / "status.json").write_text(json.dumps({"status": "completed", "epochs_completed": history[-1]["epoch"],
        "best_val_loss_epoch": history[-1]["best_val_loss_epoch"],
        "best_macro_dice_epoch": history[-1]["best_macro_dice_epoch"],
        "tie_break": "strict improvement; exact ties keep earlier epoch",
        "total_epoch_seconds": sum(row["epoch_seconds"] for row in history)}, indent=2))


if __name__ == "__main__":
    main()
