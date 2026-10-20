#!/usr/bin/env python3
"""Single-image or directory inference for the frozen four-class ONNX artifact."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steel_defect_segmentation.tiling import starts, stitch_logits


def predict(session, image_rgb):
    image = image_rgb.astype(np.float32) / 255.0
    h, w = image.shape[:2]
    positions, logits = [], []
    for top in starts(h, 256, 256):
        for left in starts(w, 512, 384):
            tile = image[top:top + 256, left:left + 512]
            if tile.shape[:2] != (256, 512):
                padded = np.zeros((256, 512, 3), dtype=np.float32)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            pred = session.run(["logits"], {"image": tile.transpose(2, 0, 1)[None]})[0][0]
            pred = pred[:, :min(256, h - top), :min(512, w - left)]
            positions.append((top, left))
            logits.append(pred)
    return stitch_logits(logits, positions, h, w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="image file or directory")
    parser.add_argument("--output", type=Path, default=Path("predictions"))
    parser.add_argument("--model", type=Path, default=Path("reports/final-delivery/e0_epoch24_best_macro.onnx"))
    parser.add_argument("--provider", choices=("cpu", "coreml"), default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.threshold != 0.5:
        parser.error("Final frozen threshold is 0.50")
    providers = ["CPUExecutionProvider"] if args.provider == "cpu" else ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.model), providers=providers)
    if args.input.is_dir():
        paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    else:
        paths = [args.input]
    if not paths:
        parser.error("No supported input images found")
    args.output.mkdir(parents=True, exist_ok=True)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        logits = predict(session, rgb)
        masks = (logits >= 0).astype(np.uint8)  # sigmoid(logit) >= 0.5
        overlay = bgr.astype(np.float32)
        for class_id, color in enumerate(colors):
            selected = masks[class_id].astype(bool)
            overlay[selected] = 0.55 * overlay[selected] + 0.45 * np.asarray(color, dtype=np.float32)
        stem = path.stem
        for class_id, mask in enumerate(masks, 1):
            cv2.imwrite(str(args.output / f"{stem}_class_{class_id}_mask.png"), mask * 255)
        cv2.imwrite(str(args.output / f"{stem}_overlay.jpg"), np.clip(overlay, 0, 255).astype(np.uint8))
        print(f"{path}: logits={logits.shape}, predicted pixels={[int(mask.sum()) for mask in masks]}")


if __name__ == "__main__":
    main()
