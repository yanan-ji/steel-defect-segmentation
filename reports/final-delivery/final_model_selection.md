# 最终模型冻结与独立测试结论

## 模型选择

最终候选冻结为 E0 的 ResNet18 U-Net Epoch 24 `best_macro_dice.pt`。选择依据是冻结验证集上正常图误检率约 34.01% 的业务参考约束下，E0 是当前候选中整体最合适者。E0 验证 Macro Dice 为 0.249202、正常图 FPR 为 34.0113%；精确值比字面 34.0100% 高 0.0013 个百分点，按项目此前使用的约 34.01% 参考口径视为近似基准，未作阈值修改。E1 的指定 Epoch 14 Macro Dice 为 0.255518，但 FPR 为 70.28%；E2 Epoch 20 Macro Dice 为 0.318016，但 FPR 为 46.33%。E2 满足严格 FPR ≤34.01% 的最佳 epoch 是 Epoch 13，Macro Dice 仅 0.161653。因此不以更高但误检显著超限的 E1/E2 分数替代 E0。

最终回归套件已在恢复的项目 `.venv` 中复跑：`20 passed`；额外的单图 ONNX 推理 smoke test 在合成 256×1600 图上生成了四个 Mask 和一个 overlay。

选择后冻结：阈值 0.50；连通域最小面积 0（不做区域过滤）；RGB 输入、float32 `/255`；不缩放原生 256×1600 图；滑窗 256×512、stride 256×384，覆盖右尾并对重叠 logits 求均值；模型输出 raw logits，阈值比较等价于 `logit >= 0`。未使用 test 指标进行选择。

## Artifact 身份

- PyTorch：`reports/baseline-bce-formal/best_macro_dice.pt`，Epoch 24，SHA-256 `fb7b5387621e9104c1513f482b48af9a04fba1c2fba3e9e362bcdb4890a5d4df`。
- ONNX：`reports/final-delivery/e0_epoch24_best_macro.onnx`，SHA-256 `27938b3ca9a79a9fe5a2e26b881014e32ee8958b36879d1b905b4bd30fab4ffe`，opset 17，约 54.4 MiB。
- Manifest hashes、依赖版本和具体指标见 `final_model_selection.json`、`dependency_versions.json`。

## 独立测试集

在模型、阈值、滑窗和 ONNX parity 均通过冻结检查后，对冻结的 1,895 张内部独立测试图进行了一次最终评估。报告见 `test_report.md`，详细聚合计数见 `test_results.json`，典型错误案例见 `test_error_cases/`。测试结果仅用于最终报告，不回流模型、阈值或后处理选择。

## ONNX 验证和速度

合成样本与冻结验证集前 8 张图完成 PyTorch/ONNX Runtime 比较；具体误差、预测一致率、后端、延迟 P50/P95 及运行环境见 `onnx_report.md` 和 `onnx_validation.json`。其中 CoreML provider 存在 CPU fallback，不代表纯 CoreML 全图执行。
