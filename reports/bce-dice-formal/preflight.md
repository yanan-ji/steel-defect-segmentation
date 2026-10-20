# E1 formal training preflight

Date: 2026-10-18 (Asia/Shanghai). E1 is initialized from scratch with seed 42.

## Frozen manifests (SHA-256)

- train: `ba7887219c3d07215f44b704f053dff6bfece91375bd7a5178353af156152c89`
- val: `593cf50fe4c05f2d788f3fb8a15a55cf2d07ae1e5644a665c81db3fe20af9470`
- test manifest hash only: `79ebfdf0244af6bf001f84de3443166fa84f2d05591a76fdea20684a66120826`

No test images or labels were used. E0 result files were left intact.

## Loss and gradient audit

For each class, Soft Dice uses the sigmoid probabilities and sums intersection and predicted/true mass across batch and spatial axes. Dice scores for classes absent from a batch are excluded from the macro loss. An all-empty batch contributes zero Dice loss while BCE remains active. The formula and derivative were checked with a positive synthetic `[2,4,8,8]` batch:

| Device | BCE | Dice | BCE gradient L1 | Dice gradient L1 | Absent-class Dice gradient L1 | Total gradient finite |
|---|---:|---:|---:|---:|---:|---|
| CPU | 0.6931473 | 0.8210526 | 0.5 | 0.1464820 | 0 | yes |
| MPS | 0.6931472 | 0.8210526 | 0.5 | 0.1464820 | 0 | yes |

For an all-empty `[1,4,3,3]` batch, Dice loss was 0 and the total gradient L1 was 0.5 on both CPU and MPS, matching BCE alone. Tests: `13 passed`. Separate reseeding with 42 produced identical initial model state tensors.

## Control variables and Pilot observation

The E0/E1 formal runner shares the model class, dataset, transforms, fixed manifests, seed, loader, optimizer, constant learning-rate scheduler, validation tiling, threshold, checkpoint and Early Stopping logic. E1 changes the objective to `BCEWithLogitsLoss + 1.0 * SoftDiceLoss`. Both stop on full-image validation BCE after six non-improving epochs, up to 30 epochs.

The earlier E1 Pilot's train Dice term rose from 0.9282 to 0.9916 while BCE fell and all validation images were predicted as background at threshold 0.5. Random crops change which classes are present in each batch; a tiny foreground region also yields a poor Soft Dice score when predictions contract toward background. This is compatible with early background prediction and is insufficient evidence of a mathematical error. The complete-data run will check Dice gradients in its first positive training batch and stop if they are nonfinite or zero.
