#!/usr/bin/env python3
"""Uniform 8-connected area filtering on frozen, full-image E0/E1 validation."""
from __future__ import annotations

import argparse
import csv
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
from steel_defect_segmentation.labels import labels_by_image, read_label_table
from steel_defect_segmentation.postprocessing import AREA_THRESHOLDS, AreaSweepAccumulator, filter_components
from analyze_thresholds import EXPERIMENTS, hash_file, infer_original, load_model, read_validation_image, reference_row


def scan_model(name, model, cfg, image_ids, labels, device, out):
    accumulator = AreaSweepAccumulator()
    cases = []
    start = time.perf_counter()
    for index, image_id in enumerate(image_ids, 1):
        image, target = read_validation_image(image_id, Path(cfg["data_root"]), cfg, labels)
        prediction = infer_original(model, image, device) >= .5
        local = accumulator.update(prediction, target)
        cases.append({"image_id": image_id, "is_normal": local["is_normal"],
                      "fp": local["fp"].tolist(), "fn": local["fn"].tolist(),
                      "removed_pixels": local["removed_pixels"].tolist(),
                      "has_prediction": local["has_prediction"].tolist()})
        if index % 100 == 0 or index == len(image_ids):
            print(f"{name}: {index}/{len(image_ids)} validation originals; {time.perf_counter()-start:.1f}s", flush=True)
    rows = accumulator.compute()
    (out / f"{name.lower()}_metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    (out / f"{name.lower()}_area_distributions.json").write_text(json.dumps(accumulator.area_distributions(), indent=2))
    (out / f"{name.lower()}_case_scores.json").write_text(json.dumps(cases))
    return rows, accumulator.area_distributions(), cases, time.perf_counter() - start


def verify_baseline(row, reference, name):
    for key in ("tp", "fp", "fn"):
        if row[key] != [int(x) for x in reference[key]]:
            raise ValueError(f"{name} unfiltered 0.50 {key} differs from formal validation")
    for key in ("macro_dice", "empty_image_false_positive_rate", "full_background_prediction_image_ratio"):
        if not np.isclose(row[key], reference[key], atol=2e-6):
            raise ValueError(f"{name} unfiltered 0.50 {key} differs from formal validation")


def plot_metric(out, results, field, filename, ylabel):
    plt.figure(figsize=(8, 5))
    for name, rows in results.items():
        plt.plot([r["minimum_area"] for r in rows], [r[field] for r in rows], marker="o", label=name)
    plt.xticks(AREA_THRESHOLDS)
    plt.xlabel("Minimum retained 8-connected area (pixels)")
    plt.ylabel(ylabel)
    plt.grid(alpha=.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / filename, dpi=160)
    plt.close()


def plot_tradeoff(out, results, ceiling):
    plt.figure(figsize=(8, 5))
    for name, rows in results.items():
        plt.plot([r["empty_image_false_positive_rate"] for r in rows],
                 [r["macro_dice"] for r in rows], marker="o", label=name)
    plt.axvline(ceiling, color="gray", ls="--", label="E0 unfiltered FPR ceiling")
    plt.xlabel("Normal-image false-positive rate")
    plt.ylabel("Macro Dice")
    plt.grid(alpha=.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "macro_dice_vs_normal_fpr.png", dpi=160)
    plt.close()


def plot_area_distributions(out, distributions):
    categories = ("normal_prediction", "defect_prediction_no_truth_overlap",
                  "defect_prediction_overlapping_truth", "true_defect")
    labels = ("1–7", "8–15", "16–31", "32–63", "64–127", "128–255", "≥256")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row_index, name in enumerate(("E0", "E1")):
        for col_index, category_pair in enumerate((categories[:2], categories[2:])):
            ax = axes[row_index, col_index]
            for category in category_pair:
                bins = np.sum([x["area_bins_1_7_to_256_plus"] for x in distributions[name][category]], axis=0)
                ax.plot(labels, bins, marker="o", label=category.replace("_", " "))
            ax.set_yscale("symlog", linthresh=1)
            ax.set_title(f"{name}: component counts")
            ax.tick_params(axis="x", rotation=35)
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "component_area_distributions.png", dpi=160)
    plt.close(fig)


def save_case(path, image, truth, prediction, class_id, area):
    base = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gt, before, after, removed = (base.copy() for _ in range(4))
    original = prediction[class_id]
    filtered = filter_components(original, area)
    gt[truth[class_id]] = (0, 255, 0)
    before[original] = (0, 0, 255)
    after[filtered] = (0, 0, 255)
    removed[original & ~filtered] = (0, 255, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.hconcat([base, gt, before, after, removed])):
        raise OSError(f"Failed to write {path}")


def save_cases(out, scores, selections, image_ids, labels, hashes, device):
    manifest = set(image_ids)
    index = []
    for name, areas in selections.items():
        model, cfg, _ = load_model(name, hashes, device)
        root = Path(cfg["data_root"])
        for area in sorted(set(areas)):
            ai = AREA_THRESHOLDS.index(area)
            chosen = []
            for c in range(4):
                for kind in ("fp", "fn"):
                    best = max(scores[name], key=lambda x: x[kind][ai][c])
                    if best[kind][ai][c] > 0:
                        chosen.append((best["image_id"], c, kind, best[kind][ai][c]))
            resolved = [x for x in scores[name] if x["is_normal"] and x["has_prediction"][0] and not x["has_prediction"][ai]]
            if resolved:
                best = max(resolved, key=lambda x: sum(x["removed_pixels"][ai]))
                c = int(np.argmax(best["removed_pixels"][ai]))
                chosen.append((best["image_id"], c, "resolved_normal_fp", sum(best["removed_pixels"][ai])))
            persistent = [x for x in scores[name] if x["is_normal"] and x["has_prediction"][ai]]
            if persistent:
                best = max(persistent, key=lambda x: sum(x["fp"][ai]))
                c = int(np.argmax(best["fp"][ai]))
                chosen.append((best["image_id"], c, "persistent_normal_fp", sum(best["fp"][ai])))
            cache = {}
            for image_id, c, kind, count in chosen:
                if image_id not in manifest:
                    raise ValueError("Case outside frozen validation manifest")
                if image_id not in cache:
                    image, truth = read_validation_image(image_id, root, cfg, labels)
                    cache[image_id] = (image, truth, infer_original(model, image, device) >= .5)
                image, truth, prediction = cache[image_id]
                relative = Path("cases") / name.lower() / f"area_{area}" / f"class_{c+1}_{kind}_{Path(image_id).stem}.jpg"
                save_case(out / relative, image, truth, prediction, c, area)
                index.append({"model": name, "minimum_area": area, "class": c+1, "kind": kind,
                              "image_id": image_id, "error_or_removed_pixels": int(count), "file": str(relative)})
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    (out / "case_index.json").write_text(json.dumps(index, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/postprocessing-analysis")
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
        raise ValueError("Frozen train/val manifest hashes changed")
    image_ids = (split_dir / "val_image_ids.txt").read_text().split()
    if len(image_ids) != 1881 or len(set(image_ids)) != 1881:
        raise ValueError("Expected 1881 unique validation originals")
    e0_cfg = yaml.safe_load((EXPERIMENTS["E0"][0] / "config.yaml").read_text())
    e1_cfg = yaml.safe_load((EXPERIMENTS["E1"][0] / "config.yaml").read_text())
    for key in ("data_root", "image_dir", "train_csv", "splits_dir", "seed"):
        if e0_cfg[key] != e1_cfg[key]:
            raise ValueError(f"Data configuration mismatch: {key}")
    labels = labels_by_image(read_label_table(Path(e0_cfg["data_root"]) / e0_cfg["train_csv"]))
    results, distributions, scores, timings = {}, {}, {}, {}
    for name in EXPERIMENTS:
        model, cfg, checkpoint = load_model(name, hashes, device)
        print(f"{name}: strict checkpoint epoch {checkpoint['epoch']} loaded on {device}", flush=True)
        rows, area_stats, cases, seconds = scan_model(name, model, cfg, image_ids, labels, device, out)
        verify_baseline(rows[0], reference_row(EXPERIMENTS[name][0], EXPERIMENTS[name][1]), name)
        results[name], distributions[name], scores[name], timings[name] = rows, area_stats, cases, seconds
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    ceiling = results["E0"][0]["empty_image_false_positive_rate"]
    feasible = [{"model": name, **row} for name, rows in results.items() for row in rows
                if row["empty_image_false_positive_rate"] <= ceiling + 1e-12]
    selected = max(feasible, key=lambda r: (r["macro_dice"], -r["empty_image_false_positive_rate"],
                                            -r["minimum_area"], r["model"] == "E0")) if feasible else None
    with (out / "area_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "minimum_area", "macro_dice", "normal_image_fpr", "full_background_ratio",
                         "normal_fp_images", "normal_images", "removed_pixels_total", "removed_components_total"] +
                        [f"class_{c}_{key}" for c in range(1, 5) for key in
                         ("dice", "iou", "precision", "recall", "tp", "fp", "fn", "removed_pixels", "removed_components")])
        for name, rows in results.items():
            for row in rows:
                writer.writerow([name, row["minimum_area"], row["macro_dice"], row["empty_image_false_positive_rate"],
                                 row["full_background_prediction_image_ratio"], row["normal_false_positive_images"],
                                 row["normal_images"], sum(row["deleted_pixels_per_class"]), sum(row["deleted_components_per_class"])] +
                                [row[key][c] for c in range(4) for key in
                                 ("dice_per_class", "iou_per_class", "precision_per_class", "recall_per_class", "tp", "fp", "fn",
                                  "deleted_pixels_per_class", "deleted_components_per_class")])
    plot_metric(out, results, "macro_dice", "macro_dice_vs_area.png", "Macro Dice")
    plot_metric(out, results, "empty_image_false_positive_rate", "normal_fpr_vs_area.png", "Normal-image false-positive rate")
    plot_tradeoff(out, results, ceiling)
    plot_area_distributions(out, distributions)
    selections = {"E0": [0, 256], "E1": [0, 256]}
    if selected:
        selections[selected["model"]].append(selected["minimum_area"])
    save_cases(out, scores, selections, image_ids, labels, hashes, device)
    summary = {"scope": "validation-tuned; full frozen 1881-image validation only; no test set read",
               "pixel_threshold": .5, "connectivity": 8, "minimum_area_candidates": AREA_THRESHOLDS,
               "manifest_hashes": hashes, "checkpoints": {name: {"path": str(folder / "best_macro_dice.pt"), "epoch": epoch}
                                                    for name, (folder, epoch) in EXPERIMENTS.items()},
               "scan_seconds": timings, "e0_unfiltered_normal_fpr_ceiling": ceiling,
               "e0_unfiltered": results["E0"][0], "e1_unfiltered": results["E1"][0],
               "best_feasible_validation_tuned": selected,
               "tie_break": "higher macro Dice, lower normal FPR, lower area threshold, E0"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    lines = ["# E0/E1 8-connected area filtering", "", summary["scope"], "",
             "Prediction uses 0.50 after overlap-averaged full-image logits. Areas are per-class and regions smaller than the threshold are removed. TP/FP/FN are recounted from each filtered full-image mask.", "",
             f"E0 unfiltered normal-image FPR ceiling: {ceiling:.6f}.", "",
             "| Model | Min area | Macro Dice | Normal-image FPR | Class 1/2/3/4 Dice | Class 1/2/3/4 Recall | Removed pixels | Removed components |",
             "|---|---:|---:|---:|---|---|---:|---:|"]
    for name, rows in results.items():
        for row in rows:
            lines.append(f"| {name} | {row['minimum_area']} | {row['macro_dice']:.6f} | {row['empty_image_false_positive_rate']:.6f} | " +
                         "/".join(f"{x:.4f}" for x in row["dice_per_class"]) + " | " +
                         "/".join(f"{x:.4f}" for x in row["recall_per_class"]) +
                         f" | {sum(row['deleted_pixels_per_class'])} | {sum(row['deleted_components_per_class'])} |")
    lines += ["", "## Selection under original E0 false-positive constraint", ""]
    if selected:
        lines.append(f"Best feasible validation-tuned row: {selected['model']} area {selected['minimum_area']}, Macro Dice {selected['macro_dice']:.6f}, normal-image FPR {selected['empty_image_false_positive_rate']:.6f}.")
    else:
        lines.append("No scanned candidate meets the E0 unfiltered normal-image FPR ceiling.")
    lines += ["", "## Error diagnosis and interpretation", ""]
    for name in ("E0", "E1"):
        normal = distributions[name]["normal_prediction"]
        component_count = sum(item["count"] for item in normal)
        tiny = sum(item["area_bins_1_7_to_256_plus"][0] for item in normal)
        under_256 = sum(sum(item["area_bins_1_7_to_256_plus"][:6]) for item in normal)
        base, strongest = results[name][0], results[name][-1]
        lines.append(
            f"- {name}: normal images contain {component_count} false-positive components; "
            f"{tiny} ({tiny/component_count:.1%}) have area <8 and {under_256} ({under_256/component_count:.1%}) have area <256. "
            f"At area 256, normal-image FPR changes {base['empty_image_false_positive_rate']:.1%}→"
            f"{strongest['empty_image_false_positive_rate']:.1%}, Macro Dice {base['macro_dice']:.4f}→"
            f"{strongest['macro_dice']:.4f}, class-1 Recall {base['recall_per_class'][0]:.4f}→"
            f"{strongest['recall_per_class'][0]:.4f}, and class-2 Recall {base['recall_per_class'][1]:.4f}→"
            f"{strongest['recall_per_class'][1]:.4f}."
        )
    true_components = distributions["E0"]["true_defect"]
    true_total = sum(item["count"] for item in true_components)
    true_small = sum(sum(item["area_bins_1_7_to_256_plus"][:6]) for item in true_components)
    lines += [
        f"- Ground-truth defect components: {true_small}/{true_total} ({true_small/true_total:.1%}) have area <256. "
        "This does not make filtering harmless: predicted components may fragment a larger true defect, and the measured class-1/2 Recall decreases confirm removed true-positive pixels.",
        "- Within the requested FPR ceiling, E0 is the practical choice. E1 has slightly higher Macro Dice throughout this area grid but much higher normal-image FPR; neither model dominates both axes, so a same-FPR ranking cannot be extrapolated outside this grid.",
        "", "## Recommendation", "",
        "Keep E0 at 0.50 with no area filter if the goal is to maximize Macro Dice under the original 34.01% FPR ceiling. "
        "A small area filter can lower E0 false alarms if a stricter FPR is required, but did not improve its Macro Dice. "
        "E1's area-256 result modestly improves Dice yet remains far above the FPR ceiling. "
        "Given persistent large false-positive components and poor class-1/2 Recall, a separately controlled E2 training study "
        "(class weighting or defect-aware sampling, one change at a time) is more promising than a large class-specific area-threshold search. "
        "Freeze its policy on validation before opening the independent test set.",
    ]
    lines += ["", "This selection is validation-tuned, not an independent test result. Class-specific area combinations were not searched.", "",
              "## Figures and cases", "", "![Macro Dice vs area](macro_dice_vs_area.png)", "",
              "![Normal FPR vs area](normal_fpr_vs_area.png)", "",
              "![Macro Dice vs normal FPR](macro_dice_vs_normal_fpr.png)", "",
              "![Component area distributions](component_area_distributions.png)", "",
              "Component distributions are also in `e0_area_distributions.json` and `e1_area_distributions.json`. Representative full-image errors are indexed by `case_index.json`."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(out / "report.md", flush=True)


if __name__ == "__main__":
    main()
