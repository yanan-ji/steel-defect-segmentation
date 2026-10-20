# E2 class-aware sampling: design and two-epoch Pilot

This is a **train-only sampling intervention** against E0. No full-data E2 training was started, no frozen manifest was changed, and the independent test set was not used.

## Controls and sampler

- Same scratch-initialized ResNet18 U-Net, seed 42, `[0,1]` inputs, random 256×512 crop, batch size 4, AdamW at `3e-4`, raw `BCEWithLogitsLoss`, full-original validation stitching, fixed 0.50 prediction threshold, and formal Early Stopping target (full validation BCE, patience 6, max 30 epochs) as E0. No Dice loss, class weights, `pos_weight`, or new augmentation.
- The sole intended formal difference is **which frozen-train source image is requested for each E0 random crop**. A train-only index records four fixed candidate-tile multi-label flags per image. Class scores add the inverse square roots of each present class's candidate-patch frequency; multi-label candidates contribute to every applicable class.
- Epoch draws preserve the exact counts in strata defined by normal-source status and the number of positive candidate tiles. Normal-source images are used once. Within defective-image strata, up to 25% of positions are exchanged in favor of higher-scoring sources; at most three uses of one image per epoch. Every use receives a fresh E0 random crop. This limits repeat intensity while retaining 8792 draws and the original normal-source/candidate-background mix. Sampling is deterministic from seed and epoch.
- The four fixed candidate tiles are a **proxy** for E0's continuous random-crop distribution. Actual cropped-mask frequencies are recorded during training and may differ. This is a sampling change, not evidence of improved validation performance.

Frozen train/val manifest SHA-256 values were checked against E0. The training-label index contains 8792 source images and 35168 candidate patches. It reproduces the audited **70.0381%** all-background candidate-patch ratio and **46.9859%** normal-source image ratio. No validation labels contribute to these weights.

| Candidate patch ratio | Original train index | Full E2 planned epoch 1 | Full E2 planned epoch 2 |
|---|---:|---:|---:|
| Background | 70.038% | 70.038% | 70.038% |
| Class 1 | 3.210% | 3.859% | 3.597% |
| Class 2 | 0.626% | 0.850% | 0.862% |
| Class 3 | 23.365% | 22.054% | 22.410% |
| Class 4 | 3.577% | 4.240% | 4.009% |

Each planned full epoch has 8792 draws, 7628 distinct source images and 1164 repeat draws; no source appears more than three times. These are **simulated full-train source draws from frozen train labels**, not measured full-training crop outcomes.

## Two-epoch Pilot: actual execution

The Pilot used 256 frozen-train originals (including prior E0 Pilot IDs and extra train-only minority-class examples) and the prior 12 frozen-validation originals. Its class composition was deliberately enriched for logic testing, so its class frequencies and Dice **must not be compared directly with E0 full-data performance**. It ran on MPS in 66.54 seconds.

| Pilot epoch | Train BCE | Val BCE | Macro Dice | Actual background crops | Actual class 1/2/3/4 crop counts | Repeat draws | Gradient |
|---:|---:|---:|---:|---:|---|---:|---|
| 1 | 0.086568 | 0.045474 | 0.000000 | 68.75% | 19 / 8 / 40 / 18 | 34 | finite |
| 2 | 0.047516 | 0.027702 | 0.000000 | 67.58% | 15 / 10 / 40 / 24 | 34 | finite |

The Pilot candidate index's own background ratio was 71.09%, and its normal-source draw ratio was preserved exactly at 46.875% in both epochs. Each epoch had 256 draws, 222 distinct source images, and at most three uses of one source. Full-image validation, finite loss/gradients and checkpoint save/load all passed. `best_val_loss.pt` selected epoch 2; the zero Macro Dice tie retained epoch 1 in `best_macro_dice.pt` under the strict-improvement rule. Zero Pilot Dice is a small-run observation, **not a finding that E2 improves or fails against E0**.

## Readiness and decision gate

The sampler, train-only patch index, deterministic bounded draw, multi-label handling, Pilot training/validation and checkpoint round-trip are working. The independent formal entry point is [`scripts/train_e2_formal.py`](../../scripts/train_e2_formal.py); it is syntax-checked but **has not run a full-data epoch or an interrupted-resume drill**. Thus E2 is ready for a user-authorized **one-full-epoch stability check**, not yet for an automatic unattended 30-epoch run. Disk free space was about 10 GiB after the Pilot, and E1 previously experienced a transient ENOSPC interruption; confirm headroom and checkpoint writes before continuing. Do not infer final performance from this Pilot.

When explicitly authorized, the formal command is:

```bash
python3 -u scripts/train_e2_formal.py --out reports/class-aware-sampling/formal
```

The formal evaluation must first use the unchanged 0.50 threshold and compare full-validation Macro Dice, four-class Dice/Precision/Recall, and normal-image false-positive rate against E0 epoch 24 (Macro Dice 0.2492, normal FPR 34.01%, class-1 Dice 0.0425, class-2 Dice 0.0272). A recall gain with materially higher false positives is **not** a comprehensive improvement. Early Stopping remains full-validation BCE, patience 6. The independent test set stays sealed until a validation policy is frozen.

Evidence: `sampling_design.json`, `train_patch_index.npz`, `pilot/config.yaml`, `pilot/metrics.jsonl`, `pilot/summary.json`, Pilot checkpoints and visualizations. All 20 current unit tests passed.
