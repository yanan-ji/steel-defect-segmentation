#!/usr/bin/env python3
"""Generate E1 learning curves and matched-validation E0 vs E1 comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def line_plot(path, epochs, series, ylabel):
    plt.figure(figsize=(9, 4))
    for label, values in series.items():
        plt.plot(epochs, values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1", default="reports/bce-dice-formal")
    parser.add_argument("--e0", default="reports/baseline-bce-formal")
    args = parser.parse_args()
    e1, e0 = Path(args.e1), Path(args.e0)
    if not (e1 / "status.json").exists():
        raise FileNotFoundError("E1 status.json is missing; training has not completed")
    e1_rows = [json.loads(line) for line in (e1 / "metrics.jsonl").read_text().splitlines() if line.strip()]
    e0_rows = [json.loads(line) for line in (e0 / "metrics.jsonl").read_text().splitlines() if line.strip()]
    e0_reference = e0_rows[23]
    if e0_reference["epoch"] != 24:
        raise ValueError("E0 reference must be epoch 24")
    epochs = [row["epoch"] for row in e1_rows]
    line_plot(e1 / "bce_curve.png", epochs, {
        "train BCE": [row["train_bce_loss"] for row in e1_rows],
        "full-image val BCE": [row["val_bce_loss"] for row in e1_rows],
    }, "BCE")
    line_plot(e1 / "dice_loss_curve.png", epochs, {
        "train Soft Dice loss": [row["train_dice_loss"] for row in e1_rows],
        "full-image val Soft Dice loss": [row["val_dice_loss"] for row in e1_rows],
    }, "Soft Dice loss")
    line_plot(e1 / "total_loss_curve.png", epochs, {
        "train BCE + Dice": [row["train_total_loss"] for row in e1_rows],
        "full-image val BCE + Dice": [row["val_total_loss"] for row in e1_rows],
    }, "E1 total loss")
    for field, title in (("dice_per_class", "Dice"), ("recall_per_class", "Recall")):
        line_plot(e1 / f"{title.lower()}_curve.png", epochs, {
            f"class {class_id}": [row[field][class_id - 1] for row in e1_rows]
            for class_id in range(1, 5)
        }, title)
    best_bce = min(e1_rows, key=lambda row: (row["val_bce_loss"], row["epoch"]))
    best_total = min(e1_rows, key=lambda row: (row["val_total_loss"], row["epoch"]))
    best_macro = max(e1_rows, key=lambda row: (row["macro_dice"], -row["epoch"]))
    lines = [
        "# E1 BCE + Dice formal experiment", "",
        "Full frozen train and validation manifests; fixed 0.5 threshold. The independent test set was not used.", "",
        "The E1 total loss has an additional Dice term and must not be compared numerically with E0 pure BCE loss.", "",
        "## E1 epochs", "",
        "| Epoch | Train BCE | Train Dice | Train total | Val BCE | Val Dice | Val total | Macro Dice | Seconds |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in e1_rows:
        lines.append("| " + " | ".join([
            str(row["epoch"]),
            *[f"{row[key]:.6f}" for key in ("train_bce_loss", "train_dice_loss", "train_total_loss", "val_bce_loss", "val_dice_loss", "val_total_loss", "macro_dice")],
            f"{row['epoch_seconds']:.1f}",
        ]) + " |")
    lines.extend([
        "", "## Checkpoint selection", "",
        f"Best validation BCE: epoch {best_bce['epoch']} ({best_bce['val_bce_loss']:.6f}).",
        f"Best validation total loss: epoch {best_total['epoch']} ({best_total['val_total_loss']:.6f}).",
        f"Best macro Dice: epoch {best_macro['epoch']} ({best_macro['macro_dice']:.6f}).",
        "Strict improvement selects a new checkpoint; exact ties retain the earlier epoch. Early Stopping monitors only full-image validation BCE.",
        "", "## Matched validation comparison", "",
        "Both rows use full validation images, whole-set TP/FP/FN, and threshold 0.5. E0 refers to its prespecified epoch 24 best-macro checkpoint; E1 refers to its best-macro checkpoint.",
        "", "| Metric | E0 epoch 24 | E1 best macro |", "|---|---:|---:|",
        f"| Macro Dice | {e0_reference['macro_dice']:.6f} | {best_macro['macro_dice']:.6f} |",
    ])
    for field, label in (("dice_per_class", "Dice"), ("recall_per_class", "Recall"), ("precision_per_class", "Precision")):
        for class_id in range(4):
            lines.append(f"| {label} class {class_id+1} | {e0_reference[field][class_id]:.6f} | {best_macro[field][class_id]:.6f} |")
    lines.extend([
        f"| Normal-image false-positive rate | {e0_reference['empty_image_false_positive_rate']:.6f} | {best_macro['empty_image_false_positive_rate']:.6f} |",
        f"| Full-background prediction ratio | {e0_reference['full_background_prediction_image_ratio']:.6f} | {best_macro['full_background_prediction_image_ratio']:.6f} |",
        f"| Actual epochs | {len(e0_rows)} | {len(e1_rows)} |",
        f"| Total epoch hours | {sum(row['epoch_seconds'] for row in e0_rows)/3600:.3f} | {sum(row['epoch_seconds'] for row in e1_rows)/3600:.3f} |",
        "", "## Curves", "",
        "![BCE curve](bce_curve.png)", "", "![Soft Dice loss curve](dice_loss_curve.png)", "",
        "![E1 total loss curve](total_loss_curve.png)", "", "![Per-class Dice curve](dice_curve.png)", "",
        "![Per-class Recall curve](recall_curve.png)", "",
        "Full per-class IoU, predicted-positive-pixel ratios, probability histograms, resource readings, and representative TP/FP/FN visualizations are in metrics.jsonl and visualizations/.",
    ])
    (e1 / "experiment_report.md").write_text("\n".join(lines) + "\n")
    print(e1 / "experiment_report.md")


if __name__ == "__main__":
    main()
