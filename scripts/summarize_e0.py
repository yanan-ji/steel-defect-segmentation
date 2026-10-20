#!/usr/bin/env python3
"""Build plots and a Markdown report from completed E0 epoch records."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="reports/baseline-bce-formal")
    args=parser.parse_args()
    out=Path(args.out)
    rows=[json.loads(line) for line in (out/"metrics.jsonl").read_text().splitlines() if line.strip()]
    if not rows: raise ValueError("No completed E0 epochs")
    epochs=[row["epoch"] for row in rows]
    plt.figure(figsize=(8,4))
    plt.plot(epochs,[r["train_loss"] for r in rows],label="train BCE")
    plt.plot(epochs,[r["val_loss"] for r in rows],label="full image val BCE")
    plt.xlabel("Epoch");plt.ylabel("BCE loss");plt.legend();plt.tight_layout();plt.savefig(out/"loss_curve.png",dpi=160);plt.close()
    for metric,label in (("dice_per_class","Dice"),("recall_per_class","Recall")):
        plt.figure(figsize=(8,4))
        for c in range(4):plt.plot(epochs,[r[metric][c] for r in rows],label=f"class {c+1}")
        plt.xlabel("Epoch");plt.ylabel(label);plt.ylim(bottom=0);plt.legend();plt.tight_layout();plt.savefig(out/f"{label.lower()}_curve.png",dpi=160);plt.close()
    best_loss=min(rows,key=lambda r:(r["val_loss"],r["epoch"]))
    best_dice=max(rows,key=lambda r:(r["macro_dice"],-r["epoch"]))
    lines=["# E0 original BCE baseline", "", "Full train and full validation manifests; official test data were not used.", "", "Selection rule: strict improvement only; exact ties retain the earliest epoch.", "", "| Epoch | Train BCE | Val BCE | Macro Dice | Dice 1 | Dice 2 | Dice 3 | Dice 4 | LR | Seconds |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:lines.append(f"| {r['epoch']} | {r['train_loss']:.6f} | {r['val_loss']:.6f} | {r['macro_dice']:.6f} | "+" | ".join(f"{v:.6f}" for v in r["dice_per_class"])+f" | {r['learning_rate']:.6g} | {r['epoch_seconds']:.1f} |")
    lines.extend(["",f"Best val loss: epoch {best_loss['epoch']} ({best_loss['val_loss']:.6f}).",f"Best macro Dice: epoch {best_dice['epoch']} ({best_dice['macro_dice']:.6f}).", "", "Full precision, recall, IoU, TP/FP/FN, probability histograms, resource statistics, and per epoch examples are in metrics.jsonl and visualizations/.", "", "![Loss curve](loss_curve.png)", "", "![Dice curve](dice_curve.png)", "", "![Recall curve](recall_curve.png)"])
    (out/"experiment_report.md").write_text("\n".join(lines)+"\n")
    print(out/"experiment_report.md")
if __name__=="__main__":main()
