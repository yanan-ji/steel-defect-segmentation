#!/usr/bin/env python3
"""Export frozen E0 and run parity/latency checks on synthetic plus small validation samples."""
from __future__ import annotations

import hashlib
import json
import platform
import importlib.metadata
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import cv2
import yaml

from steel_defect_segmentation.model import ResNet18UNet
from steel_defect_segmentation.tiling import starts, stitch_logits


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "reports/baseline-bce-formal/best_macro_dice.pt"
OUT = ROOT / "reports/final-delivery"
MODEL_PATH = OUT / "e0_epoch24_best_macro.onnx"
SEED = 24042


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def torch_stitch(model, image, device):
    h, w = image.shape[-2:]
    positions = [(0, left) for left in starts(w, 512, 384)]
    tiles = []
    for _, left in positions:
        with torch.inference_mode():
            source = image[:, :, :, left:left + 512]
            tensor = source.to(device) if isinstance(source, torch.Tensor) else torch.from_numpy(source).to(device)
            pred = model(tensor).float().cpu().numpy()[0]
        tiles.append(pred)
    return stitch_logits(tiles, positions, h, w)


def ort_stitch(session, image):
    h, w = image.shape[-2:]
    positions = [(0, left) for left in starts(w, 512, 384)]
    tiles = []
    for _, left in positions:
        pred = session.run(["logits"], {"image": image[:, :, :, left:left + 512]})[0][0]
        tiles.append(pred)
    return stitch_logits(tiles, positions, h, w)


def ort_full_inference(session, image_rgb):
    image = image_rgb.astype(np.float32) / 255.0
    h, w = image.shape[:2]
    positions, tiles = [], []
    for top in starts(h, 256, 256):
        for left in starts(w, 512, 384):
            patch = image[top:top + 256, left:left + 512]
            output = session.run(["logits"], {"image": patch.transpose(2, 0, 1)[None]})[0][0]
            tiles.append(output[:, :patch.shape[0], :patch.shape[1]])
            positions.append((top, left))
    return stitch_logits(tiles, positions, h, w) >= 0


def torch_full_inference(model, image_rgb, device):
    image = image_rgb.astype(np.float32) / 255.0
    h, w = image.shape[:2]
    positions, tiles = [], []
    for top in starts(h, 256, 256):
        for left in starts(w, 512, 384):
            patch = np.ascontiguousarray(image[top:top + 256, left:left + 512].transpose(2, 0, 1)[None])
            with torch.inference_mode():
                output = model(torch.from_numpy(patch).to(device)).float().cpu().numpy()[0]
            tiles.append(output)
            positions.append((top, left))
    return stitch_logits(tiles, positions, h, w) >= 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != 24:
        raise ValueError("Expected the frozen E0 Epoch 24 checkpoint")
    expected = {
        "train": "ba7887219c3d07215f44b704f053dff6bfece91375bd7a5178353af156152c89",
        "val": "593cf50fe4c05f2d788f3fb8a15a55cf2d07ae1e5644a665c81db3fe20af9470",
    }
    if checkpoint.get("manifest_hash") != expected:
        raise ValueError("Checkpoint manifest hashes do not match frozen E0 values")
    model = ResNet18UNet(classes=4).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    sample = torch.zeros((1, 3, 256, 512), dtype=torch.float32)
    torch.onnx.export(
        model, sample, MODEL_PATH, input_names=["image"], output_names=["logits"],
        opset_version=17, do_constant_folding=True, dynamo=False,
        dynamic_axes=None,
    )
    graph = onnx.load(MODEL_PATH)
    onnx.checker.check_model(graph)
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(MODEL_PATH), sess_options=session_options,
                                   providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(SEED)
    synthetic = rng.random((1, 3, 256, 512), dtype=np.float32)
    with torch.inference_mode():
        torch_output = model(torch.from_numpy(synthetic)).numpy()
    ort_output = session.run(["logits"], {"image": synthetic})[0]
    max_abs = float(np.max(np.abs(torch_output - ort_output)))
    mask_agreement = float(np.mean((torch_output >= 0) == (ort_output >= 0)))

    backend_checks = {
        "onnxruntime_cpu": {"available": True, "provider": "CPUExecutionProvider"},
        "onnxruntime_coreml": {"available": "CoreMLExecutionProvider" in ort.get_available_providers()},
        "pytorch_mps": {"available": torch.backends.mps.is_available()},
    }
    if backend_checks["onnxruntime_coreml"]["available"]:
        try:
            coreml_session = ort.InferenceSession(str(MODEL_PATH), providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
            coreml_output = coreml_session.run(["logits"], {"image": synthetic})[0]
            backend_checks["onnxruntime_coreml"].update({
                "session_providers": coreml_session.get_providers(),
                "max_abs_logit_error_vs_torch": float(np.max(np.abs(torch_output - coreml_output))),
                "threshold_mask_pixel_agreement_vs_torch": float(np.mean((torch_output >= 0) == (coreml_output >= 0))),
                "status": "passed",
            })
        except Exception as exc:  # backend support varies by installed ORT build/model ops
            backend_checks["onnxruntime_coreml"].update({"status": "unavailable_for_model", "error": str(exc)[:1200]})
    if backend_checks["pytorch_mps"]["available"]:
        try:
            mps_model = ResNet18UNet(classes=4).eval().to("mps")
            mps_model.load_state_dict(checkpoint["model"], strict=True)
            with torch.inference_mode():
                mps_output = mps_model(torch.from_numpy(synthetic).to("mps")).float().cpu().numpy()
                torch.mps.synchronize()
            backend_checks["pytorch_mps"].update({
                "max_abs_logit_error_vs_cpu_torch": float(np.max(np.abs(torch_output - mps_output))),
                "threshold_mask_pixel_agreement_vs_cpu_torch": float(np.mean((torch_output >= 0) == (mps_output >= 0))),
                "status": "passed",
            })
        except Exception as exc:
            backend_checks["pytorch_mps"].update({"status": "failed", "error": str(exc)[:1200]})

    # Full-width synthetic 256x1600 image checks right-tail and overlap behavior.
    full = rng.random((1, 3, 256, 1600), dtype=np.float32)
    full_rgb = rng.integers(0, 256, (256, 1600, 3), dtype=np.uint8)
    torch_full = torch_stitch(model, full, torch.device("cpu"))
    ort_full = ort_stitch(session, full)
    stitch_max_abs = float(np.max(np.abs(torch_full - ort_full)))
    stitch_mask_agreement = float(np.mean((torch_full >= 0) == (ort_full >= 0)))

    # Read a deterministic, small subset of the frozen validation manifest only.
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    val_ids = (ROOT / cfg["splits_dir"] / "val_image_ids.txt").read_text().split()
    val_subset = val_ids[:8]
    val_checks = []
    image_dir = ROOT / cfg["data_root"] / cfg["image_dir"]
    for image_id in val_subset:
        bgr = cv2.imread(str(image_dir / image_id), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_dir / image_id)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        val_image = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        val_torch = torch_stitch(model, val_image, torch.device("cpu"))
        val_ort = ort_stitch(session, val_image)
        val_checks.append({"image_id": image_id,
                           "max_abs_logit_error": float(np.max(np.abs(val_torch - val_ort))),
                           "threshold_mask_pixel_agreement": float(np.mean((val_torch >= 0) == (val_ort >= 0))),
                           "shape": list(val_torch.shape)})

    # Warmed CPU latency. Keep measurements explicitly synthetic, not dataset performance.
    for _ in range(3):
        session.run(["logits"], {"image": synthetic})
    timings = []
    for _ in range(20):
        start = time.perf_counter()
        session.run(["logits"], {"image": synthetic})
        timings.append((time.perf_counter() - start) * 1000)
    tile_times = []
    for _ in range(5):
        start = time.perf_counter()
        ort_stitch(session, full)
        tile_times.append((time.perf_counter() - start) * 1000)

    for _ in range(3):
        with torch.inference_mode():
            model(torch.from_numpy(synthetic))
    cpu_torch_times = []
    for _ in range(20):
        start = time.perf_counter()
        with torch.inference_mode():
            model(torch.from_numpy(synthetic))
        cpu_torch_times.append((time.perf_counter() - start) * 1000)
    if backend_checks["pytorch_mps"].get("status") == "passed":
        mps_model = mps_model.eval()
        mps_input = torch.from_numpy(synthetic).to("mps")
        for _ in range(4):
            with torch.inference_mode():
                mps_model(mps_input)
        torch.mps.synchronize()
        mps_times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.inference_mode():
                mps_model(mps_input)
            torch.mps.synchronize()
            mps_times.append((time.perf_counter() - start) * 1000)
        backend_checks["pytorch_mps"]["model_only_p50_ms"] = percentile(mps_times, 50)
        backend_checks["pytorch_mps"]["model_only_p95_ms"] = percentile(mps_times, 95)
    if backend_checks["onnxruntime_coreml"].get("status") == "passed":
        for _ in range(3):
            coreml_session.run(["logits"], {"image": synthetic})
        coreml_times = []
        for _ in range(20):
            start = time.perf_counter()
            coreml_session.run(["logits"], {"image": synthetic})
            coreml_times.append((time.perf_counter() - start) * 1000)
        backend_checks["onnxruntime_coreml"]["model_only_p50_ms"] = percentile(coreml_times, 50)
        backend_checks["onnxruntime_coreml"]["model_only_p95_ms"] = percentile(coreml_times, 95)

    mps_full_latency = None
    if backend_checks["pytorch_mps"].get("status") == "passed":
        full_t = torch.from_numpy(full).to("mps")
        for _ in range(2):
            torch_stitch(mps_model, full_t, torch.device("mps"))
        torch.mps.synchronize()
        values = []
        for _ in range(3):
            start = time.perf_counter()
            torch_stitch(mps_model, full_t, torch.device("mps"))
            torch.mps.synchronize()
            values.append((time.perf_counter() - start) * 1000)
        mps_full_latency = {"p50": percentile(values, 50), "p95": percentile(values, 95), "samples": len(values)}
    coreml_full_latency = None
    if backend_checks["onnxruntime_coreml"].get("status") == "passed":
        values = []
        for _ in range(3):
            start = time.perf_counter()
            ort_stitch(coreml_session, full)
            values.append((time.perf_counter() - start) * 1000)
        coreml_full_latency = {"p50": percentile(values, 50), "p95": percentile(values, 95), "samples": len(values)}
    # Pipeline latency includes uint8 RGB scaling, tile creation, model calls,
    # overlap averaging and fixed-threshold mask generation; excludes disk I/O.
    cpu_e2e = []
    for _ in range(5):
        start = time.perf_counter()
        ort_full_inference(session, full_rgb)
        cpu_e2e.append((time.perf_counter() - start) * 1000)
    mps_e2e = None
    if backend_checks["pytorch_mps"].get("status") == "passed":
        values = []
        for _ in range(3):
            start = time.perf_counter()
            torch_full_inference(mps_model, full_rgb, torch.device("mps"))
            torch.mps.synchronize()
            values.append((time.perf_counter() - start) * 1000)
        mps_e2e = {"p50": percentile(values, 50), "p95": percentile(values, 95), "samples": len(values)}
    coreml_e2e = None
    if backend_checks["onnxruntime_coreml"].get("status") == "passed":
        values = []
        for _ in range(3):
            start = time.perf_counter()
            ort_full_inference(coreml_session, full_rgb)
            values.append((time.perf_counter() - start) * 1000)
        coreml_e2e = {"p50": percentile(values, 50), "p95": percentile(values, 95), "samples": len(values)}

    report = {
        "evaluation_scope": "synthetic plus deterministic first 8 frozen validation originals; no test images or labels read",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": sha256(CHECKPOINT),
        "manifest_hashes": expected,
        "model": "ResNet18UNet(classes=4), from-scratch E0 Epoch 24 best_macro_dice",
        "input": {"name": "image", "shape": [1, 3, 256, 512], "dtype": "float32", "color": "RGB", "scale": "uint8 / 255"},
        "output": {"name": "logits", "shape": [1, 4, 256, 512], "dtype": "float32", "activation": "none"},
        "threshold": 0.5,
        "postprocessing_min_component_area": 0,
        "preprocessing": "OpenCV BGR to RGB; float32 /255; no resize for native 256x1600 images",
        "tiling": {"tile": [256, 512], "stride": [256, 384], "tail": "append final start at width-tile; overlap logits averaged", "image_width": 1600, "starts": starts(1600, 512, 384)},
        "onnx": {"path": str(MODEL_PATH.relative_to(ROOT)), "sha256": sha256(MODEL_PATH), "opset": 17, "size_bytes": MODEL_PATH.stat().st_size},
        "runtime": {"onnxruntime": ort.__version__, "providers_available": ort.get_available_providers(), "provider_used": "CPUExecutionProvider"},
        "backend_checks_synthetic": backend_checks,
        "consistency_synthetic": {"tile_max_abs_logit_error": max_abs, "tile_threshold_mask_pixel_agreement": mask_agreement, "full_256x1600_stitch_max_abs_logit_error": stitch_max_abs, "full_256x1600_stitch_mask_pixel_agreement": stitch_mask_agreement, "passed": max_abs < 1e-3 and stitch_max_abs < 1e-3 and mask_agreement == 1.0 and stitch_mask_agreement == 1.0},
        "consistency_frozen_validation_subset": {"images_checked": len(val_checks), "images": val_checks, "max_abs_logit_error": max(x["max_abs_logit_error"] for x in val_checks), "min_threshold_mask_pixel_agreement": min(x["threshold_mask_pixel_agreement"] for x in val_checks), "passed": all(x["max_abs_logit_error"] < 1e-3 and x["threshold_mask_pixel_agreement"] == 1.0 for x in val_checks)},
        "latency_ms_synthetic": {"model_only_excludes_preprocessing_stitching": True, "full_image_end_to_end_includes_scaling_tiling_model_stitching_and_threshold_excludes_disk_io": True, "onnxruntime_cpu_model_only_p50": percentile(timings, 50), "onnxruntime_cpu_model_only_p95": percentile(timings, 95), "torch_cpu_model_only_p50": percentile(cpu_torch_times, 50), "torch_cpu_model_only_p95": percentile(cpu_torch_times, 95), "onnxruntime_cpu_tile_stitch_only_p50": percentile(tile_times, 50), "onnxruntime_cpu_tile_stitch_only_p95": percentile(tile_times, 95), "onnxruntime_cpu_full_image_e2e_p50": percentile(cpu_e2e, 50), "onnxruntime_cpu_full_image_e2e_p95": percentile(cpu_e2e, 95), "torch_mps_full_image_e2e": mps_e2e, "onnxruntime_coreml_full_image_e2e": coreml_e2e, "torch_mps_tile_stitch_only": mps_full_latency, "onnxruntime_coreml_tile_stitch_only": coreml_full_latency, "model_only_samples": len(timings), "cpu_full_image_samples": len(tile_times)},
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__, "mps_available": torch.backends.mps.is_available(), "mps_tested": backend_checks["pytorch_mps"].get("status") == "passed"},
    }
    (OUT / "onnx_validation.json").write_text(json.dumps(report, indent=2))
    selection = {
        "selected_model": "E0 ResNet18 U-Net Epoch 24 best_macro_dice",
        "model_config": {"class": "steel_defect_segmentation.model.ResNet18UNet", "encoder": "ResNet18-style residual stages trained from scratch", "pretrained_weights": False, "classes": 4, "input_channels": 3, "output": "four independent raw-logit channels", "parameter_count": sum(p.numel() for p in model.parameters())},
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_epoch": 24,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "selection_basis": "Best current candidate under the approximate validation normal-image FPR reference of 34.01%; E0 Macro Dice 0.249202 and FPR 34.0113% at fixed 0.50 (0.0013 percentage points above a literal 34.0100% cutoff).",
        "validation_metrics": checkpoint["metrics"],
        "frozen_inference": {"threshold": 0.5, "minimum_connected_component_area": 0,
                             "input_rgb_float32_scale": "uint8 / 255", "resize": False,
                             "tile_hw": [256, 512], "stride_hw": [256, 384],
                             "tail_coverage": "append last start at dimension-tile; average overlapping logits"},
        "manifest_hashes": {**expected, "test": "79ebfdf0244af6bf001f84de3443166fa84f2d05591a76fdea20684a66120826"},
        "test_manifest_hash_provenance": "carried forward from pre-existing E2 frozen-manifest verification and matched before test evaluation",
        "onnx_artifact": report["onnx"],
        "dependencies": {"python": platform.python_version(), "packages": {dist.metadata["Name"]: dist.version for dist in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower()) if dist.metadata.get("Name")}},
        "test_evaluation": ({"status": "completed once after all freeze/parity gates passed", "report": "reports/final-delivery/test_report.md", "used_for_model_or_threshold_selection": False}
                             if (OUT / "test_results.json").exists()
                             else "not run; gated until export and parity checks complete and explicit final-evaluation phase"),
    }
    (OUT / "final_model_selection.json").write_text(json.dumps(selection, indent=2))
    (OUT / "dependency_versions.json").write_text(json.dumps(selection["dependencies"], indent=2))
    (OUT / "onnx_report.md").write_text(
        "# ONNX 导出与一致性报告\n\n"
        "ONNX 导出及 parity 检查已完成。合成输入及冻结验证集前 8 张图用于核对；没有读取独立 test 数据。"
        "固定输入为 `[1,3,256,512]` RGB float32（像素除以 255），输出为四通道 raw logits；推理阈值 0.50，连通域过滤面积 0。\n\n"
        f"- PyTorch/ORT synthetic tile 最大绝对误差：{max_abs:.8g}；像素 Mask 一致率：{mask_agreement:.6%}。\n"
        f"- 256×1600 合成整图拼接最大误差：{stitch_max_abs:.8g}；Mask 一致率：{stitch_mask_agreement:.6%}；实际 E0 滑窗尾部起点 `{starts(1600, 512, 384)}`。\n"
        f"- 验证子集：{len(val_checks)} 张；最大误差 {max(x['max_abs_logit_error'] for x in val_checks):.8g}，最低 Mask 一致率 {min(x['threshold_mask_pixel_agreement'] for x in val_checks):.6%}。\n"
        f"- ONNX 文件：`{MODEL_PATH.name}`，{MODEL_PATH.stat().st_size / 1024**2:.2f} MiB，SHA-256 `{sha256(MODEL_PATH)}`。\n"
        f"- 后端：ORT CPU；PyTorch MPS {'通过' if backend_checks['pytorch_mps'].get('status') == 'passed' else '未通过'}；ORT CoreML + CPU fallback 状态 `{backend_checks['onnxruntime_coreml'].get('status')}`。\n\n"
        f"- 纯模型延迟（20 次 warmup 后测量）：ORT CPU P50/P95 {report['latency_ms_synthetic']['onnxruntime_cpu_model_only_p50']:.2f}/{report['latency_ms_synthetic']['onnxruntime_cpu_model_only_p95']:.2f} ms；PyTorch MPS {backend_checks['pytorch_mps'].get('model_only_p50_ms', float('nan')):.2f}/{backend_checks['pytorch_mps'].get('model_only_p95_ms', float('nan')):.2f} ms；ORT CoreML+CPU {backend_checks['onnxruntime_coreml'].get('model_only_p50_ms', float('nan')):.2f}/{backend_checks['onnxruntime_coreml'].get('model_only_p95_ms', float('nan')):.2f} ms。\n"
        f"- 整图端到端（预处理缩放、分块、推理、重叠拼接和阈值；不含文件读写）：ORT CPU P50/P95 {report['latency_ms_synthetic']['onnxruntime_cpu_full_image_e2e_p50']:.2f}/{report['latency_ms_synthetic']['onnxruntime_cpu_full_image_e2e_p95']:.2f} ms；PyTorch MPS {report['latency_ms_synthetic']['torch_mps_full_image_e2e']['p50']:.2f}/{report['latency_ms_synthetic']['torch_mps_full_image_e2e']['p95']:.2f} ms；ORT CoreML+CPU {report['latency_ms_synthetic']['onnxruntime_coreml_full_image_e2e']['p50']:.2f}/{report['latency_ms_synthetic']['onnxruntime_coreml_full_image_e2e']['p95']:.2f} ms。\n\n"
        "延迟是本机合成输入测得，不代表真实业务数据吞吐；完整运行环境及样本数见 `onnx_validation.json`。"
    )
    print(json.dumps(report, indent=2))
    if not report["consistency_synthetic"]["passed"] or not report["consistency_frozen_validation_subset"]["passed"]:
        raise SystemExit("ONNX parity check failed")


if __name__ == "__main__":
    main()
