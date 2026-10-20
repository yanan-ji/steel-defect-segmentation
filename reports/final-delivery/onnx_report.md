# ONNX 导出与一致性报告

ONNX 导出及 parity 检查已完成。合成输入及冻结验证集前 8 张图用于核对；没有读取独立 test 数据。固定输入为 `[1,3,256,512]` RGB float32（像素除以 255），输出为四通道 raw logits；推理阈值 0.50，连通域过滤面积 0。

- PyTorch/ORT synthetic tile 最大绝对误差：3.0517578e-05；像素 Mask 一致率：100.000000%。
- 256×1600 合成整图拼接最大误差：2.4795532e-05；Mask 一致率：100.000000%；实际 E0 滑窗尾部起点 `[0, 384, 768, 1088]`。
- 验证子集：8 张；最大误差 0.00019836426，最低 Mask 一致率 100.000000%。
- ONNX 文件：`e0_epoch24_best_macro.onnx`，54.45 MiB，SHA-256 `27938b3ca9a79a9fe5a2e26b881014e32ee8958b36879d1b905b4bd30fab4ffe`。
- 后端：ORT CPU；PyTorch MPS 通过；ORT CoreML + CPU fallback 状态 `passed`。

- 纯模型延迟（20 次 warmup 后测量）：ORT CPU P50/P95 69.05/71.42 ms；PyTorch MPS 28.77/31.11 ms；ORT CoreML+CPU 15.69/16.64 ms。
- 整图端到端（预处理缩放、分块、推理、重叠拼接和阈值；不含文件读写）：ORT CPU P50/P95 302.58/313.75 ms；PyTorch MPS 130.56/152.18 ms；ORT CoreML+CPU 68.99/73.14 ms。

延迟是本机合成输入测得，不代表真实业务数据吞吐；完整运行环境及样本数见 `onnx_validation.json`。