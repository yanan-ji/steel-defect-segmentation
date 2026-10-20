#!/usr/bin/env python3
"""One-time final evaluation of the frozen E0 ONNX artifact on the held-out split."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from steel_defect_segmentation.labels import decode_multilabel, labels_by_image, read_label_table
from steel_defect_segmentation.metrics import ConfusionAccumulator
from steel_defect_segmentation.tiling import stitch_logits, tile_image
from train_e0_formal import draw_case


OUT = ROOT / "reports/final-delivery"
EXPECTED_TEST_SHA = "79ebfdf0244af6bf001f84de3443166fa84f2d05591a76fdea20684a66120826"
EXPECTED_CKPT_SHA = "fb7b5387621e9104c1513f482b48af9a04fba1c2fba3e9e362bcdb4890a5d4df"
EXPECTED_ONNX_SHA = "27938b3ca9a79a9fe5a2e26b881014e32ee8958b36879d1b905b4bd30fab4ffe"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    # Verify selection/export gates before opening the held-out manifest or images.
    validation = json.loads((OUT / "onnx_validation.json").read_text())
    selection = json.loads((OUT / "final_model_selection.json").read_text())
    if selection["checkpoint_sha256"] != EXPECTED_CKPT_SHA:
        raise ValueError("Frozen E0 checkpoint identity changed")
    if validation["onnx"]["sha256"] != EXPECTED_ONNX_SHA or not validation["consistency_synthetic"]["passed"] or not validation["consistency_frozen_validation_subset"]["passed"]:
        raise ValueError("ONNX export/parity gate has not passed")
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    split_dir = ROOT / cfg["splits_dir"]
    test_path = split_dir / "test_image_ids.txt"
    if sha256(test_path) != EXPECTED_TEST_SHA:
        raise ValueError("Frozen test manifest hash mismatch")
    test_ids = test_path.read_text().split()
    if len(test_ids) != 1895 or len(set(test_ids)) != len(test_ids):
        raise ValueError(f"Expected 1895 unique frozen test images; found {len(test_ids)}")

    root = ROOT / cfg["data_root"]
    label_table = read_label_table(root / cfg["train_csv"])
    labels = labels_by_image(label_table[label_table.ImageId.isin(test_ids)])
    session = ort.InferenceSession(str(OUT / "e0_epoch24_best_macro.onnx"), providers=["CPUExecutionProvider"])
    accumulator = ConfusionAccumulator(classes=4)
    empty_count = empty_fp_count = background_count = 0
    total_pixels = 0
    examples = {c: {"fn": None, "fp": None, "tp": None} for c in range(4)}
    case_dir = OUT / "test_error_cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for index, image_id in enumerate(test_ids, 1):
        bgr = cv2.imread(str(root / cfg["image_dir"] / image_id), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(root / cfg["image_dir"] / image_id)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if (height, width) != (256, 1600):
            raise ValueError(f"Unexpected frozen test image size {image_id}: {(height, width)}")
        target = decode_multilabel(labels.get(image_id, {}), height, width).astype(np.uint8)
        positions, tile_logits = [], []
        for top, left, patch in tile_image(image, 256, 512, 256, 384):
            inp = np.ascontiguousarray(patch.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
            tile_logits.append(session.run(["logits"], {"image": inp})[0][0])
            positions.append((top, left))
        logits_np = stitch_logits(tile_logits, positions, height, width)
        logits = torch.from_numpy(logits_np).unsqueeze(0)
        truth = torch.from_numpy(target.astype(np.float32)).unsqueeze(0)
        accumulator.update(logits, truth, .5)
        probabilities = torch.sigmoid(logits)[0].numpy()
        prediction = probabilities >= .5
        empty = not target.any()
        empty_count += int(empty)
        empty_fp_count += int(empty and prediction.any())
        background_count += int(not prediction.any())
        total_pixels += height * width
        for class_id in range(4):
            gt = target[class_id].astype(bool)
            if examples[class_id]["fn"] is None and np.any(gt & ~prediction[class_id]):
                examples[class_id]["fn"] = image_id
                draw_case(case_dir / f"class_{class_id+1}_fn_{image_id}.jpg", image, target, probabilities, class_id)
            if examples[class_id]["fp"] is None and np.any(~gt & prediction[class_id]):
                examples[class_id]["fp"] = image_id
                draw_case(case_dir / f"class_{class_id+1}_fp_{image_id}.jpg", image, target, probabilities, class_id)
            if examples[class_id]["tp"] is None and np.any(gt & prediction[class_id]):
                examples[class_id]["tp"] = image_id
                draw_case(case_dir / f"class_{class_id+1}_tp_{image_id}.jpg", image, target, probabilities, class_id)
        if index % 200 == 0:
            print(f"final ONNX test evaluation {index}/{len(test_ids)}", flush=True)

    metrics = accumulator.compute()
    metrics.update({
        "split": "frozen internal held-out test; evaluated once after model/threshold/tiling/ONNX freeze",
        "test_images": len(test_ids), "normal_images": empty_count,
        "normal_image_false_positive_images": empty_fp_count,
        "empty_image_false_positive_rate": empty_fp_count / empty_count if empty_count else None,
        "full_background_prediction_images": background_count,
        "full_background_prediction_image_ratio": background_count / len(test_ids),
        "prediction_positive_pixel_ratio_per_class": [(tp + fp) / total_pixels for tp, fp in zip(accumulator.tp.tolist(), accumulator.fp.tolist())],
        "threshold": .5, "minimum_connected_component_area": 0,
        "inference_backend": "ONNX Runtime CPUExecutionProvider",
        "elapsed_seconds": time.perf_counter() - started,
        "representative_error_cases": examples,
        "checkpoint_sha256": EXPECTED_CKPT_SHA, "onnx_sha256": EXPECTED_ONNX_SHA,
        "test_manifest_sha256": EXPECTED_TEST_SHA,
    })
    (OUT / "test_results.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    selection["test_evaluation"] = {
        "status": "completed once after all freeze/parity gates passed",
        "report": "reports/final-delivery/test_report.md",
        "used_for_model_or_threshold_selection": False,
    }
    (OUT / "final_model_selection.json").write_text(json.dumps(selection, indent=2))
    lines = [
        "# 最终独立测试集结果", "",
        "这是冻结 E0 Epoch 24 best-macro ONNX 模型在冻结内部独立测试清单上的一次最终评估。评估前已固定 checkpoint、ONNX artifact、阈值 0.50、连通域面积 0、预处理和滑窗拼接；结果不用于回选或调参。", "",
        f"- 图像数：{len(test_ids)}；正常图：{empty_count}；正常图误检图像：{empty_fp_count}。",
        f"- Macro Dice：{metrics['macro_dice']:.6f}",
        f"- 正常图误检率：{metrics['empty_image_false_positive_rate']:.2%}",
        f"- 全背景预测比例：{metrics['full_background_prediction_image_ratio']:.2%}", "",
        "| 类别 | Dice | IoU | Precision | Recall | TP | FP | FN |", "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in range(4):
        lines.append(f"| {c+1} | {metrics['dice_per_class'][c]:.6f} | {metrics['iou_per_class'][c]:.6f} | {metrics['precision_per_class'][c]:.6f} | {metrics['recall_per_class'][c]:.6f} | {metrics['tp'][c]:.0f} | {metrics['fp'][c]:.0f} | {metrics['fn'][c]:.0f} |")
    lines.extend(["", f"完整指标 JSON：`test_results.json`；按类别保存的典型 FN/FP/TP 可视化：`test_error_cases/`。推理耗时 {metrics['elapsed_seconds']:.1f} 秒，后端 ONNX Runtime CPU。", ""])
    (OUT / "test_report.md").write_text("\n".join(lines))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
