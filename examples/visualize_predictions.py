#!/usr/bin/env python3
"""Render a qualitative panel for frozen-model predictions: input | GT | prediction | error.

The script runs the frozen ONNX artefact through the exact inference contract
documented in the README (RGB, float32 / 255, 256x512 windows, stride 256/384,
overlap-averaged logits, threshold 0.50, no connected-component filtering) and
draws ground truth from ``train.csv`` next to the prediction. Nothing is
synthesised: every mask comes from the frozen model, and every GT mask is
decoded from the official RLE annotations.

Panels per case (2x2 grid):
    top-left     original image
    top-right    ground truth (per-class colour overlay)
    bottom-left  prediction at threshold 0.50
    bottom-right error map: green = TP, red = FP, blue = FN (union over classes)

Usage
-----
    python examples/visualize_predictions.py \
        --ids 861a262c8.jpg 16a50624d.jpg 05b847b1f.jpg \
        --out docs/figures/qualitative_predictions.png

Requires the base stack (numpy, opencv-python, pandas, PyYAML) plus
onnxruntime; PyTorch is not required.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steel_defect_segmentation.labels import decode_multilabel, labels_by_image, read_label_table
from steel_defect_segmentation.tiling import starts, stitch_logits

TILE_H, TILE_W, STRIDE_H, STRIDE_W = 256, 512, 256, 384
CLASS_COLORS = [(255, 0, 255), (255, 255, 0), (0, 0, 255), (0, 165, 255)]  # BGR, class 1..4
HEADER_H = 36
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_session(model_path: Path, provider: str) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"] if provider == "cpu" else [
        "CoreMLExecutionProvider", "CPUExecutionProvider"
    ]
    return ort.InferenceSession(str(model_path), providers=providers)


def predict_probabilities(session: ort.InferenceSession, image_rgb: np.ndarray) -> np.ndarray:
    """Tiled inference -> sigmoid probabilities of shape [4, H, W]."""
    image = image_rgb.astype(np.float32) / 255.0
    height, width = image.shape[:2]
    positions, tiles = [], []
    for top in starts(height, TILE_H, STRIDE_H):
        for left in starts(width, TILE_W, STRIDE_W):
            patch = image[top:top + TILE_H, left:left + TILE_W]
            if patch.shape[:2] != (TILE_H, TILE_W):
                padded = np.zeros((TILE_H, TILE_W, 3), dtype=np.float32)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded
            logits = session.run(["logits"], {"image": patch.transpose(2, 0, 1)[None]})[0][0]
            tiles.append(logits[:, :min(TILE_H, height - top), :min(TILE_W, width - left)])
            positions.append((top, left))
    stitched = stitch_logits(tiles, positions, height, width)
    # Numerically stable sigmoid; extreme logits are saturated far outside the 0.50 decision boundary.
    return 1.0 / (1.0 + np.exp(-np.clip(stitched, -60.0, 60.0)))


def overlay_classes(bgr: np.ndarray, masks: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = bgr.astype(np.float32)
    for class_index, color in enumerate(CLASS_COLORS):
        selected = masks[class_index].astype(bool)
        if selected.any():
            out[selected] = (1 - alpha) * out[selected] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def error_panel(bgr: np.ndarray, prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.float32)
    gray = gray * 0.45
    predicted_any = prediction.any(axis=0)
    truth_any = truth.any(axis=0)
    for selected, color in (
        (predicted_any & truth_any, (60, 200, 60)),    # true positive (green)
        (predicted_any & ~truth_any, (50, 50, 235)),   # false positive (red)
        (~predicted_any & truth_any, (235, 120, 40)),  # false negative (blue)
    ):
        if selected.any():
            gray[selected] = 0.25 * gray[selected] + 0.75 * np.asarray(color, dtype=np.float32)
    return np.clip(gray, 0, 255).astype(np.uint8)


def with_header(panel: np.ndarray, text: str) -> np.ndarray:
    header = np.full((HEADER_H, panel.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, text, (10, 26), LABEL_FONT, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    return np.vstack([header, panel])


def build_case(bgr: np.ndarray, truth: np.ndarray, prediction: np.ndarray, title: str) -> np.ndarray:
    """Vertical 4-row strip: input, ground truth, prediction, error map (full 1600 px width)."""
    return np.vstack([
        with_header(bgr, f"{title} - input"),
        with_header(overlay_classes(bgr, truth), f"{title} - ground truth (class 1..4 colour)"),
        with_header(overlay_classes(bgr, prediction), f"{title} - prediction @0.50, no area filter"),
        with_header(error_panel(bgr, prediction, truth), f"{title} - error map (any class): green TP, red FP, blue FN"),
    ])


def per_class_dice(prediction: np.ndarray, truth: np.ndarray) -> list[float]:
    scores = []
    for class_index in range(prediction.shape[0]):
        pred = prediction[class_index].astype(bool)
        gt = truth[class_index].astype(bool)
        denominator = pred.sum() + gt.sum()
        scores.append(1.0 if denominator == 0 else float(2 * np.logical_and(pred, gt).sum() / denominator))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="+", required=True, help="image file names, e.g. 861a262c8.jpg")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "reports/final-delivery/e0_epoch24_best_macro.onnx")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/figures/qualitative_predictions.png")
    parser.add_argument("--provider", choices=("cpu", "coreml"), default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.threshold != 0.5:
        parser.error("The frozen inference threshold is 0.50")
    if not args.model.exists():
        parser.error(f"ONNX model not found: {args.model}. See README 'Model weights'.")

    cfg = yaml.safe_load(args.config.read_text())
    data_root = ROOT / cfg["data_root"]
    image_dir = data_root / cfg["image_dir"]
    label_table = read_label_table(data_root / cfg["train_csv"])
    labels = labels_by_image(label_table[label_table.ImageId.isin(args.ids)])
    session = load_session(args.model, args.provider)

    cases = []
    for image_id in args.ids:
        bgr = cv2.imread(str(image_dir / image_id), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_dir / image_id)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        truth = decode_multilabel(labels.get(image_id, {}), *bgr.shape[:2]).astype(bool)
        probabilities = predict_probabilities(session, rgb)
        prediction = probabilities >= args.threshold
        dice = per_class_dice(prediction, truth)
        cases.append(build_case(bgr, truth, prediction, image_id))
        print(f"{image_id}: per-class Dice {[round(x, 4) for x in dice]} "
              f"| predicted px {[int(m.sum()) for m in prediction]} "
              f"| GT px {[int(m.sum()) for m in truth]}")

    figure = np.vstack(cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), figure)
    print(f"wrote {args.out} ({figure.shape[1]}x{figure.shape[0]})")


if __name__ == "__main__":
    main()
