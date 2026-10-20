#!/usr/bin/env python3
"""Validation-only E0/E1 uniform threshold sweep; never reads the test set."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steel_defect_segmentation.labels import decode_multilabel, labels_by_image, read_label_table
from steel_defect_segmentation.model import ResNet18UNet
from steel_defect_segmentation.thresholds import THRESHOLDS, ThresholdAccumulator
from steel_defect_segmentation.tiling import stitch_logits, tile_image


EXPERIMENTS = {
    "E0": (Path("reports/baseline-bce-formal"), 24),
    "E1": (Path("reports/bce-dice-formal"), 14),
}


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_validation_image(image_id, root, cfg, labels):
    path = root / cfg["image_dir"] / image_id
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if image.shape != (256, 1600, 3):
        raise ValueError(f"Unexpected validation image shape for {image_id}: {image.shape}")
    target = decode_multilabel(labels.get(image_id, {}), 256, 1600).astype(bool)
    return image, target


def infer_original(model, image, device):
    positions, parts = [], []
    with torch.inference_mode():
        for top, left, patch in tile_image(image, 256, 512, 256, 384):
            tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))).float().div_(255).unsqueeze(0).to(device)
            parts.append(model(tensor)[0].cpu().numpy())
            positions.append((top, left))
    logits = stitch_logits(parts, positions, 256, 1600)
    if logits.shape != (4, 256, 1600) or not np.isfinite(logits).all():
        raise ValueError("Invalid stitched original-image logits")
    # Apply sigmoid after overlap-averaging logits, exactly as formal validation.
    return torch.sigmoid(torch.from_numpy(logits)).numpy()


def load_model(experiment, hashes, device):
    folder, epoch = EXPERIMENTS[experiment]
    cfg = yaml.safe_load((folder / "config.yaml").read_text())
    if cfg["training"]["threshold"] != .5 or cfg["data_root"] != "severstal-steel-defect-detection":
        raise ValueError(f"Unexpected {experiment} data/evaluation configuration")
    saved_hashes = json.loads((folder / "manifest_hashes.json").read_text())
    if saved_hashes != hashes:
        raise ValueError(f"{experiment} manifest hashes differ from frozen files")
    checkpoint = torch.load(folder / "best_macro_dice.pt", map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != epoch or checkpoint["manifest_hash"] != hashes:
        raise ValueError(f"{experiment} checkpoint epoch/hash mismatch")
    if checkpoint.get("objective", "bce") != ("bce" if experiment == "E0" else "bce-dice"):
        raise ValueError(f"{experiment} objective mismatch")
    model = ResNet18UNet(classes=4)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, cfg, checkpoint


def scan_model(model, cfg, image_ids, labels, device, out, name):
    root = Path(cfg["data_root"])
    accumulator = ThresholdAccumulator(THRESHOLDS)
    cases = []
    started = time.perf_counter()
    for index, image_id in enumerate(image_ids, 1):
        image, target = read_validation_image(image_id, root, cfg, labels)
        probabilities = infer_original(model, image, device)
        local = accumulator.update(probabilities, target)
        cases.append({"image_id": image_id, "is_normal": local["is_normal"],
                      "max_probability": local["max_probability"],
                      "fp": local["fp"].tolist(), "fn": local["fn"].tolist()})
        if index % 100 == 0 or index == len(image_ids):
            print(f"{name}: {index}/{len(image_ids)} validation images, elapsed {time.perf_counter()-started:.1f}s", flush=True)
    rows = accumulator.compute()
    (out / f"{name.lower()}_metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    (out / f"{name.lower()}_case_scores.json").write_text(json.dumps(cases))
    return rows, cases, time.perf_counter() - started


def reference_row(folder, epoch):
    rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines() if line.strip()]
    return next(row for row in rows if row["epoch"] == epoch)


def verify_original_half(row, reference, model):
    for key in ("macro_dice", "empty_image_false_positive_rate", "full_background_prediction_image_ratio"):
        if not np.isclose(row[key], reference[key], atol=2e-6):
            raise ValueError(f"{model} threshold 0.50 {key} differs from formal validation: {row[key]} vs {reference[key]}")
    for key in ("tp", "fp", "fn"):
        if row[key] != [int(x) for x in reference[key]]:
            raise ValueError(f"{model} threshold 0.50 {key} differs from formal validation")


def plot_curves(out, results, field, filename, ylabel):
    plt.figure(figsize=(8, 5))
    for name, rows in results.items():
        plt.plot([r["threshold"] for r in rows], [r[field] for r in rows], marker="o", label=name)
    plt.axvline(.5, color="gray", ls="--", lw=1, label="original 0.50")
    plt.xlabel("Uniform threshold")
    plt.ylabel(ylabel)
    plt.grid(alpha=.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / filename, dpi=160)
    plt.close()


def plot_tradeoff(out, results, feasible_fpr):
    plt.figure(figsize=(8, 5))
    for name, rows in results.items():
        x = [r["empty_image_false_positive_rate"] for r in rows]
        y = [r["macro_dice"] for r in rows]
        plt.plot(x, y, marker="o", label=name)
        half = next(r for r in rows if np.isclose(r["threshold"], .5))
        plt.scatter([half["empty_image_false_positive_rate"]], [half["macro_dice"]], s=85, marker="x")
    plt.axvline(feasible_fpr, color="gray", ls="--", lw=1, label="E0 original FPR ceiling")
    plt.xlabel("Normal-image false-positive rate")
    plt.ylabel("Macro Dice")
    plt.grid(alpha=.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "macro_dice_vs_normal_fpr.png", dpi=160)
    plt.close()


def plot_per_class(out, results):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for c, ax in enumerate(axes.flat):
        for name, rows in results.items():
            ax.plot([r["threshold"] for r in rows], [r["dice_per_class"][c] for r in rows], marker=".", label=name)
        ax.set_title(f"Class {c+1}")
        ax.grid(alpha=.25)
        ax.set_ylabel("Dice")
        ax.set_xlabel("Threshold")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out / "per_class_dice_vs_threshold.png", dpi=160)
    plt.close(fig)


def save_case(path, image, target, probabilities, class_id, threshold):
    base = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    truth, prediction = base.copy(), base.copy()
    truth[target[class_id]] = (0, 255, 0)
    prediction[probabilities[class_id] >= threshold] = (0, 0, 255)
    heat = cv2.applyColorMap(np.uint8(np.clip(probabilities[class_id], 0, 1) * 255), cv2.COLORMAP_JET)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.hconcat([base, truth, prediction, heat])):
        raise OSError(f"Failed to save {path}")


def save_representative_cases(out, results, case_scores, selections, image_ids, labels, device):
    manifest_set = set(image_ids)
    case_index = []
    for name, thresholds in selections.items():
        model, cfg, _ = load_model(name, results["hashes"], device)
        root = Path(cfg["data_root"])
        scores = case_scores[name]
        for threshold in sorted(set(thresholds)):
            ti = int(np.where(np.isclose(THRESHOLDS, threshold))[0][0])
            chosen = []
            for c in range(4):
                for kind in ("fp", "fn"):
                    best = max(scores, key=lambda s: s[kind][ti][c])
                    if best[kind][ti][c] > 0:
                        chosen.append((best["image_id"], c, kind, best[kind][ti][c]))
            normal = [s for s in scores if s["is_normal"] and s["max_probability"] >= threshold]
            if normal:
                best = max(normal, key=lambda s: sum(s["fp"][ti]))
                c = int(np.argmax(best["fp"][ti]))
                chosen.append((best["image_id"], c, "normal_fp", sum(best["fp"][ti])))
            cache = {}
            for image_id, c, kind, pixels in chosen:
                if image_id not in manifest_set:
                    raise ValueError("Case image is outside frozen validation manifest")
                if image_id not in cache:
                    image, target = read_validation_image(image_id, root, cfg, labels)
                    cache[image_id] = (image, target, infer_original(model, image, device))
                image, target, probabilities = cache[image_id]
                relative = Path("cases") / name.lower() / f"threshold_{threshold:.2f}" / f"class_{c+1}_{kind}_{image_id}.jpg"
                save_case(out / relative, image, target, probabilities, c, threshold)
                case_index.append({"model": name, "threshold": threshold, "class": c+1,
                                   "kind": kind, "image_id": image_id, "error_pixels": int(pixels), "file": str(relative)})
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    (out / "case_index.json").write_text(json.dumps(case_index, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/threshold-analysis")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; choose --device cpu explicitly")
    device = torch.device(args.device)
    split_dir = Path("data/processed/splits")
    hashes = {key: hash_file(split_dir / f"{key}_image_ids.txt") for key in ("train", "val")}
    expected = {"train": "ba7887219c3d07215f44b704f053dff6bfece91375bd7a5178353af156152c89",
                "val": "593cf50fe4c05f2d788f3fb8a15a55cf2d07ae1e5644a665c81db3fe20af9470"}
    if hashes != expected:
        raise ValueError("Frozen train/val manifests changed")
    image_ids = (split_dir / "val_image_ids.txt").read_text().split()
    if len(image_ids) != 1881 or len(set(image_ids)) != 1881:
        raise ValueError("Expected 1881 unique frozen validation images")
    e0_cfg = yaml.safe_load((EXPERIMENTS["E0"][0] / "config.yaml").read_text())
    e1_cfg = yaml.safe_load((EXPERIMENTS["E1"][0] / "config.yaml").read_text())
    for key in ("data_root", "image_dir", "train_csv", "splits_dir", "seed"):
        if e0_cfg[key] != e1_cfg[key]:
            raise ValueError(f"E0/E1 data config mismatch: {key}")
    labels = labels_by_image(read_label_table(Path(e0_cfg["data_root"]) / e0_cfg["train_csv"]))
    results, case_scores, timings = {}, {}, {}
    for name in EXPERIMENTS:
        model, cfg, checkpoint = load_model(name, hashes, device)
        print(f"{name}: checkpoint epoch {checkpoint['epoch']} loaded strictly on {device}", flush=True)
        rows, cases, seconds = scan_model(model, cfg, image_ids, labels, device, out, name)
        half = next(row for row in rows if np.isclose(row["threshold"], .5))
        verify_original_half(half, reference_row(EXPERIMENTS[name][0], EXPERIMENTS[name][1]), name)
        results[name], case_scores[name], timings[name] = rows, cases, seconds
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    ceiling = next(r for r in results["E0"] if np.isclose(r["threshold"], .5))["empty_image_false_positive_rate"]
    candidates = [{"model": name, **row} for name, rows in results.items() for row in rows
                  if row["empty_image_false_positive_rate"] <= ceiling + 1e-12]
    # Deterministic tie-break: macro Dice, lower normal FPR, then lower threshold, then E0.
    selected = max(candidates, key=lambda r: (r["macro_dice"], -r["empty_image_false_positive_rate"],
                                               -r["threshold"], r["model"] == "E0")) if candidates else None
    with (out / "threshold_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["model", "threshold", "macro_dice", "normal_image_fpr", "full_background_ratio", "normal_fp_images", "normal_images"]
        header += [f"class_{c}_{metric}" for c in range(1, 5) for metric in ("dice", "iou", "precision", "recall", "tp", "fp", "fn")]
        writer.writerow(header)
        for name, rows in results.items():
            for row in rows:
                writer.writerow([name, f"{row['threshold']:.2f}", row["macro_dice"], row["empty_image_false_positive_rate"],
                                 row["full_background_prediction_image_ratio"], row["normal_false_positive_images"], row["normal_images"]] +
                                [row[key][c] for c in range(4) for key in ("dice_per_class", "iou_per_class", "precision_per_class", "recall_per_class", "tp", "fp", "fn")])
    plot_curves(out, results, "macro_dice", "macro_dice_vs_threshold.png", "Macro Dice")
    plot_curves(out, results, "empty_image_false_positive_rate", "normal_fpr_vs_threshold.png", "Normal-image false-positive rate")
    plot_tradeoff(out, results, ceiling)
    plot_per_class(out, results)
    # Save original operating points and each model's unconstrained Dice peak
    # as diagnostic cases; only the constrained selection is a recommendation.
    selections = {name: [.5, max(rows, key=lambda r: r["macro_dice"])["threshold"]]
                  for name, rows in results.items()}
    if selected:
        selections[selected["model"]].append(selected["threshold"])
    save_representative_cases(out, {"hashes": hashes}, case_scores, selections, image_ids, labels, device)
    summary = {"scope": "validation-tuned; frozen 1881-image validation only; no test images read",
               "thresholds": THRESHOLDS.tolist(), "manifest_hashes": hashes,
               "checkpoints": {name: {"path": str(folder / "best_macro_dice.pt"), "epoch": epoch}
                               for name, (folder, epoch) in EXPERIMENTS.items()},
               "device": str(device), "scan_seconds": timings, "e0_original_normal_fpr_ceiling": ceiling,
               "e0_original": next(r for r in results["E0"] if np.isclose(r["threshold"], .5)),
               "e1_original": next(r for r in results["E1"] if np.isclose(r["threshold"], .5)),
               "unconstrained_macro_peaks": {name: max(rows, key=lambda r: r["macro_dice"])
                                             for name, rows in results.items()},
               "best_feasible_validation_tuned": selected,
               "tie_break": "higher macro Dice, lower normal FPR, lower threshold, E0"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    lines = ["# E0/E1 uniform threshold analysis", "", summary["scope"], "",
             "Sigmoid follows full-image overlap-averaged logits. All confusion counts are accumulated over the entire validation set. The 0.50 rows reproduce the formal evaluations.", "",
             f"E0 original 0.50 normal-image FPR ceiling: {ceiling:.6f}.", "",
             "| Model | Threshold | Macro Dice | Normal-image FPR | Full-background ratio | Class 1/2/3/4 Dice |",
             "|---|---:|---:|---:|---:|---|" ]
    important = [("E0 original", summary["e0_original"]), ("E1 original", summary["e1_original"])]
    if selected:
        important.append((f"{selected['model']} validation-tuned, feasible", selected))
    for name, row in important:
        lines.append(f"| {name} | {row['threshold']:.2f} | {row['macro_dice']:.6f} | {row['empty_image_false_positive_rate']:.6f} | {row['full_background_prediction_image_ratio']:.6f} | " +
                     "/".join(f"{x:.4f}" for x in row["dice_per_class"]) + " |")
    if not selected:
        lines += ["", "No scanned configuration satisfies the E0 original normal-image FPR ceiling."]
    lines += ["", "Selection uses validation only and is not an independent test estimate. Uniform thresholds are the only tuned parameter; no class-specific thresholds, postprocessing or training were used.", "",
              "## Curves", "", "![Macro Dice vs threshold](macro_dice_vs_threshold.png)", "",
              "![Normal FPR vs threshold](normal_fpr_vs_threshold.png)", "",
              "![Macro Dice vs normal FPR](macro_dice_vs_normal_fpr.png)", "",
              "![Per-class Dice vs threshold](per_class_dice_vs_threshold.png)", "",
              "Complete class-wise Dice, IoU, Precision, Recall and TP/FP/FN are in threshold_metrics.csv; representative cases are indexed in case_index.json."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(out / "report.md", flush=True)


if __name__ == "__main__":
    main()
