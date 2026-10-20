# Epoch 6 interruption and recovery

At approximately 2026-10-18 20:35 China time, the first attempt at Epoch 6 stopped after validating 600 of 1,881 images. The traceback was `OSError: [Errno 28] No space left on device` when `run.log` attempted to `fsync`. There was no nonfinite loss, gradient, or model error. The incomplete Epoch 6 has no metrics row or checkpoint.

The complete Epoch 5 `last.pt` was verified to contain model, optimizer, constant scheduler, Early Stopping state, crop/DataLoader/Python/NumPy/PyTorch RNG states, manifest hashes, and metrics matching the five completed JSONL rows. Available disk space later measured about 15 GiB; no E0 or E1 output files were deleted.

The first resume attempt exposed a recovery-code issue: loading the checkpoint directly to MPS also moved the CPU RNG ByteTensor to MPS. The loader was corrected to load the checkpoint on CPU, after which model and optimizer state tensors are placed on MPS by their state loaders. A non-training restore check passed (`exp_avg` and `exp_avg_sq` on MPS; CPU RNG restored), and all 13 tests passed. Training restarted at Epoch 6 from the complete Epoch 5 checkpoint at 2026-10-18 21:35 China time. The interrupted partial Epoch 6 validation is excluded from all reported metrics.
