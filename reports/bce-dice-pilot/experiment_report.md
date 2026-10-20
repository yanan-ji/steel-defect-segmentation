# E1: BCE + Soft Dice Pilot

This was a two-epoch logic check on the same fixed representative E0 Pilot IDs: 16 original train images and 12 independent original validation images. Both subsets include classes 1–4 and normal images. Validation used complete 256×1600 images with overlapping tile logits stitched before applying a fixed 0.5 threshold. The official test set was not used.

## Controlled configuration

The only intended change from E0 is the training objective: `BCEWithLogitsLoss + 1.0 × SoftDiceLoss`. Model, from-scratch seed 42 initialization, batch size 4, AdamW learning rate 3e-4, random crop sampling, [0,1] image scaling, full-image validation, and 0.5 metric threshold were retained. Formal E1 Early Stopping should monitor full validation BCE with patience 6, as in E0. This two-epoch Pilot did not invoke Early Stopping.

Soft Dice uses `sigmoid(logits)` and computes each class across the current batch. Classes with no positive target pixels in that batch are excluded from the Dice average; when all four are absent, Dice Loss is differentiable zero and BCE supplies the background penalty. Validation total loss is the mean of image-level `BCE + Dice` on stitched full-image logits; it is diagnostic and does not replace validation BCE as the planned stopping criterion.

## Actual results

| Epoch | Train BCE | Train Dice | Train total | Val BCE | Val Dice | Val total | Macro Dice |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.514706 | 0.928226 | 1.442932 | 0.351279 | 0.557296 | 0.908574 | 0.000000 |
| 2 | 0.074765 | 0.991645 | 1.066410 | 0.100251 | 0.583165 | 0.683416 | 0.000000 |

At both epochs, each class had Dice, IoU, Precision and Recall of zero, and each predicted-positive-pixel ratio was zero. All 12 validation images were predicted as entirely background. The normal-image false-positive rate was zero. The validation target contained 4,132 / 6,420 / 64,886 / 29,840 positive pixels in classes 1–4; all were false negatives. These values demonstrate an early all-background tendency under this small Pilot, not comparative performance against the full-data E0 experiment.

The MPS Pilot ran for 12.05 seconds. Current MPS allocated memory at completion was 549,208,576 bytes. Loss and gradients remained finite. The training logic and checkpoint load were verified; unit tests: 11 passed.

## Selection and next step

`best_val_bce.pt` and `best_val_total.pt` selected epoch 2. Macro Dice tied at zero, so the predefined strict-improvement rule retained epoch 1 in `best_macro_dice.pt`. `last.pt` is epoch 2. Full metrics and sample IDs are in `metrics.json`, `pilot_train_ids.txt`, and `pilot_val_ids.txt`.

For a formal E1 run, use the complete frozen train and validation manifests, max 30 epochs, batch size 4, AdamW at 3e-4, E1 objective above, fixed 0.5 threshold, and Early Stopping on full-image validation BCE with patience 6. Preserve separate best-validation-BCE, best-validation-total-loss and best-macro-Dice checkpoints and complete resume state. Review this Pilot before starting the formal run; no formal E1 training was started here.
