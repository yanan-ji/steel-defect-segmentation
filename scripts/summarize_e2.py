#!/usr/bin/env python3
"""Summarize completed E2 training and matched validation comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


E0_DIR = Path("reports/baseline-bce-formal")
E1_DIR = Path("reports/bce-dice-formal")
E2_DIR = Path("reports/class-aware-sampling-formal/epoch1")
REPORT_DIR = Path("reports/class-aware-sampling-formal")
FPR_LIMIT = 0.3401


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_line(path: Path, epochs, series, ylabel, ylim=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, values in series.items():
        ax.plot(epochs, values, marker="o", markersize=3, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plot_dir = REPORT_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)
    e0_rows = rows(E0_DIR / "metrics.jsonl")
    e1_rows = rows(E1_DIR / "metrics.jsonl")
    e2_rows = rows(E2_DIR / "metrics.jsonl")
    status = json.loads((E2_DIR / "status.json").read_text())
    if status.get("status") != "completed" or status.get("epochs_completed") != len(e2_rows):
        raise ValueError("E2 status does not match the number of completed metric rows")
    if [r["epoch"] for r in e2_rows] != list(range(1, len(e2_rows) + 1)):
        raise ValueError("E2 epochs are not contiguous from the restored Epoch 1")

    e0 = next(row for row in e0_rows if row["epoch"] == 24)
    e1 = next(row for row in e1_rows if row["epoch"] == 14)
    e2_macro = max(e2_rows, key=lambda row: (row["macro_dice"], -row["epoch"]))
    e2_bce = min(e2_rows, key=lambda row: (row["val_loss"], row["epoch"]))
    constrained = [row for row in e2_rows if row["empty_image_false_positive_rate"] <= FPR_LIMIT]
    e2_constrained = max(constrained, key=lambda row: (row["macro_dice"], -row["epoch"])) if constrained else None

    # Confirm the selected artifacts are readable and their saved epochs match the logs.
    checkpoint_epochs = {}
    for name, expected_epoch in (("best_val_loss.pt", status["best_val_loss_epoch"]),
                                 ("best_macro_dice.pt", status["best_macro_dice_epoch"]),
                                 ("last.pt", status["epochs_completed"])):
        checkpoint = torch.load(E2_DIR / name, map_location="cpu", weights_only=False)
        if checkpoint.get("epoch") != expected_epoch:
            raise ValueError(f"{name} epoch does not match status.json")
        if checkpoint.get("manifest_hashes", {}).get("train") != "ba7887219c3d07215f44b704f053dff6bfece91375bd7a5178353af156152c89":
            raise ValueError(f"{name} has unexpected train manifest hash")
        if checkpoint.get("manifest_hashes", {}).get("val") != "593cf50fe4c05f2d788f3fb8a15a55cf2d07ae1e5644a665c81db3fe20af9470":
            raise ValueError(f"{name} has unexpected val manifest hash")
        if checkpoint.get("manifest_hashes", {}).get("test") != "79ebfdf0244af6bf001f84de3443166fa84f2d05591a76fdea20684a66120826":
            raise ValueError(f"{name} has unexpected test manifest hash")
        checkpoint_epochs[name] = checkpoint["epoch"]

    epochs = [row["epoch"] for row in e2_rows]
    save_line(plot_dir / "bce_curve.png", epochs, {
        "train BCE": [row["train_loss"] for row in e2_rows],
        "full-image val BCE": [row["val_loss"] for row in e2_rows],
    }, "BCE loss")
    save_line(plot_dir / "macro_dice_curve.png", epochs, {
        "Macro Dice": [row["macro_dice"] for row in e2_rows],
    }, "Macro Dice", (0, 1))
    for metric, label in (("dice_per_class", "dice"), ("recall_per_class", "recall")):
        save_line(plot_dir / f"{label}_by_class.png", epochs, {
            f"class {class_id + 1}": [row[metric][class_id] for row in e2_rows]
            for class_id in range(4)
        }, label.title(), (0, 1))
    save_line(plot_dir / "false_positive_and_background.png", epochs, {
        "normal-image false-positive rate": [row["empty_image_false_positive_rate"] for row in e2_rows],
        "all-background prediction ratio": [row["full_background_prediction_image_ratio"] for row in e2_rows],
    }, "Image ratio", (0, 1))
    save_line(plot_dir / "sampling_distribution.png", epochs, {
        "background crop ratio": [row["sampling"]["actual_random_crop_background_ratio"] for row in e2_rows],
        **{f"class {c+1} positive crop ratio": [row["sampling"]["actual_random_crop_class_ratio"][c] for row in e2_rows]
           for c in range(4)},
    }, "Train crop ratio", (0, 1))

    fig, ax = plt.subplots(figsize=(7, 5))
    for row in e2_rows:
        ax.scatter(row["empty_image_false_positive_rate"], row["macro_dice"], s=22, alpha=.7,
                   color="#3478b8")
        ax.annotate(str(row["epoch"]), (row["empty_image_false_positive_rate"], row["macro_dice"]),
                    xytext=(3, 2), textcoords="offset points", fontsize=7)
    ax.axvline(FPR_LIMIT, color="#bd3d3a", linestyle="--", label="E0 FPR reference 34.01%")
    ax.axhline(e0["macro_dice"], color="#555555", linestyle=":", label="E0 Macro Dice 0.249202")
    ax.set_xlabel("Normal-image false-positive rate")
    ax.set_ylabel("Macro Dice")
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "macro_dice_fpr_tradeoff.png", dpi=160)
    plt.close(fig)

    comparisons = [("E0 Epoch 24 best macro", e0), ("E1 Epoch 14 best macro", e1),
                   (f"E2 Epoch {e2_macro['epoch']} best macro", e2_macro)]
    if e2_constrained:
        comparisons.append((f"E2 Epoch {e2_constrained['epoch']} best under FPR limit", e2_constrained))
    report = [
        "# E2 类别感知采样正式训练：验证集结果与 E0/E1 对照", "",
        "训练从完整 E2 Epoch 1 checkpoint 恢复，随后从 Epoch 2 开始；没有重复计算 Epoch 1。总训练 22 epochs，在完整验证 BCE 连续 6 轮未改善后于 Epoch 22 Early Stop。未读取独立 test 集；test manifest 仅用于哈希核对。所有模型性能均为冻结验证集 validation 结果，阈值固定 0.50。", "",
        "E0/E1/E2 对照使用相同完整验证清单、整图拼接、全验证集 TP/FP/FN 累计和阈值 0.50。E0 采用指定 Epoch 24，E1 采用指定 Epoch 14，E2 主结果采用验证 Macro Dice 最大的 epoch。E2 的验证集结果不能当作独立测试或泛化性能。", "",
        "## 主要判断", "",
        f"E2 best-macro checkpoint 为 Epoch {e2_macro['epoch']}：Macro Dice {e2_macro['macro_dice']:.6f}，正常图误检率 {e2_macro['empty_image_false_positive_rate']:.2%}。它超过 E0 的 Macro Dice {e0['macro_dice']:.6f}，但 FPR 高于 34.01% 约束，因此**没有满足联合目标**。",
    ]
    if e2_constrained:
        report.append(f"在正常图 FPR ≤34.01% 的 E2 epoch 中，最高 Macro Dice 是 Epoch {e2_constrained['epoch']}：{e2_constrained['macro_dice']:.6f}，FPR {e2_constrained['empty_image_false_positive_rate']:.2%}；低于 E0 的 {e0['macro_dice']:.6f}。故 E2 也未在误检约束内胜过 E0。")
    else:
        report.append("E2 没有任何 epoch 达到正常图误检率 ≤34.01%，因此约束内不存在候选。")
    report.extend([
        "因此按预先设定的判断标准，E0 保留为最终候选。E2 的高 Macro Dice 候选伴随较高正常图误检，不能描述为全面提升。此处不追加新实验。", "",
        "## E0 / E1 / E2 指标对照", "",
        "| 模型/验证 checkpoint | Epoch | Macro Dice | 正常图误检率 | 全背景预测率 |", "|---|---:|---:|---:|---:|",
    ])
    for name, row in comparisons:
        report.append(f"| {name} | {row['epoch']} | {row['macro_dice']:.6f} | {row['empty_image_false_positive_rate']:.2%} | {row['full_background_prediction_image_ratio']:.2%} |")
    report.extend(["", "| 类别 | 指标 | E0 Epoch 24 | E1 Epoch 14 | E2 best Macro Epoch %d |" % e2_macro["epoch"],
                   "|---:|---|---:|---:|---:|"])
    for class_id in range(4):
        for field, label in (("dice_per_class", "Dice"), ("iou_per_class", "IoU"),
                             ("precision_per_class", "Precision"), ("recall_per_class", "Recall")):
            report.append(f"| {class_id+1} | {label} | {e0[field][class_id]:.6f} | {e1[field][class_id]:.6f} | {e2_macro[field][class_id]:.6f} |")
    report.extend(["", "## E2 逐 Epoch 记录", "",
                   "| Epoch | Train BCE | Val BCE | Macro Dice | Dice 1/2/3/4 | Recall 1/2/3/4 | Normal FPR | All-bg ratio | Train bg crop | Stale | Seconds |",
                   "|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|"])
    for row in e2_rows:
        report.append(f"| {row['epoch']} | {row['train_loss']:.6f} | {row['val_loss']:.6f} | {row['macro_dice']:.6f} | "
                      + "/".join(f"{v:.3f}" for v in row["dice_per_class"]) + " | "
                      + "/".join(f"{v:.3f}" for v in row["recall_per_class"]) + " | "
                      + f"{row['empty_image_false_positive_rate']:.2%} | {row['full_background_prediction_image_ratio']:.2%} | "
                      + f"{row['sampling']['actual_random_crop_background_ratio']:.2%} | {row['early_stopping_stale']} | {row['epoch_seconds']:.1f} |")
    total_seconds = sum(row["epoch_seconds"] for row in e2_rows)
    report.extend(["", "## Checkpoint 与资源", "",
                   f"- Best validation BCE：Epoch {status['best_val_loss_epoch']}，Val BCE {e2_bce['val_loss']:.6f}。",
                   f"- Best Macro Dice：Epoch {status['best_macro_dice_epoch']}，Macro Dice {e2_macro['macro_dice']:.6f}。",
                   f"- 实际完成：{len(e2_rows)} epochs / 30 上限；Early Stopping 于 Epoch {len(e2_rows)} 触发。",
                   f"- 累计 epoch 训练+验证耗时：{total_seconds/3600:.2f} 小时（不含进程启动等外围时间）。",
                   f"- Checkpoint 核验 epoch：`best_val_loss.pt`={checkpoint_epochs['best_val_loss.pt']}，`best_macro_dice.pt`={checkpoint_epochs['best_macro_dice.pt']}，`last.pt`={checkpoint_epochs['last.pt']}；三者均可加载，manifest 哈希匹配。",
                   "- Checkpoint 中保留模型、optimizer、scheduler、RNG、采样计划、Early Stopping 状态和冻结 manifest hashes。",
                   "- 全量 per-class 指标、TP/FP/FN、采样比例、耗时、MPS 内存、典型漏检/误检记录见 `epoch1/metrics.jsonl`；日志见 `epoch1/run.log`。", "",
                   "## 错误与采样分析", "",
                   f"E2 Macro Dice 随 epoch 变化并伴随正常图 FPR 波动：最佳 Macro Dice Epoch {e2_macro['epoch']} 的 FPR 为 {e2_macro['empty_image_false_positive_rate']:.2%}；最优 BCE 在 Epoch {e2_bce['epoch']}。两者优化口径不同。少数类召回虽随训练出现，但类别 1/2 仍是主要短板；不能只凭 Macro Dice 或仅凭 FPR 宣称整体胜出。实际采样分布每轮记录于 metrics.jsonl，候选 tile 频率不等于实际裁剪后 Mask 类别比例。", "",
                   "## 曲线", "",
                   "![Train and validation BCE](plots/bce_curve.png)", "",
                   "![Macro Dice](plots/macro_dice_curve.png)", "",
                   "![Per-class Dice](plots/dice_by_class.png)", "",
                   "![Per-class Recall](plots/recall_by_class.png)", "",
                   "![Macro Dice and FPR tradeoff](plots/macro_dice_fpr_tradeoff.png)", "",
                   "![Sampling distribution](plots/sampling_distribution.png)", "",
                   "## 最终决定", "",
                   "E2 未在正常图误检率 ≤34.01% 时超过 E0 的 Macro Dice 0.249202。按要求保留 E0 为最终候选；不运行独立测试、不部署，也不自动启动额外实验。"])
    report_path = REPORT_DIR / "experiment_report.md"
    report_path.write_text("\n".join(report) + "\n")

    summary = {"epochs_completed": len(e2_rows), "early_stopped": status["epochs_completed"] < 30,
               "best_val_loss_epoch": status["best_val_loss_epoch"], "best_macro_dice_epoch": status["best_macro_dice_epoch"],
               "best_macro_dice": e2_macro["macro_dice"], "best_macro_fpr": e2_macro["empty_image_false_positive_rate"],
               "best_under_fpr_epoch": e2_constrained["epoch"] if e2_constrained else None,
               "best_under_fpr_macro_dice": e2_constrained["macro_dice"] if e2_constrained else None,
               "best_under_fpr_fpr": e2_constrained["empty_image_false_positive_rate"] if e2_constrained else None,
               "e0_reference_macro_dice": e0["macro_dice"], "e0_reference_fpr": e0["empty_image_false_positive_rate"],
               "joint_target_met": bool(e2_constrained and e2_constrained["macro_dice"] > e0["macro_dice"]),
               "test_set_used": False, "checkpoint_epochs": checkpoint_epochs,
               "total_epoch_seconds": total_seconds, "report": str(report_path)}
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
