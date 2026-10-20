"""Streaming, exact threshold statistics for four-label original images."""
from __future__ import annotations

import numpy as np


THRESHOLDS = np.arange(10, 96, 5, dtype=np.float64) / 100.0


class ThresholdAccumulator:
    """Accumulate whole-dataset pixel confusion without retaining probability maps.

    A score equal to a threshold is positive. Jointly empty class confusion has
    Dice/IoU zero at dataset level, matching ConfusionAccumulator.compute().
    """

    def __init__(self, thresholds=THRESHOLDS, classes=4):
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        if self.thresholds.ndim != 1 or not len(self.thresholds) or not np.all(np.diff(self.thresholds) > 0):
            raise ValueError("Thresholds must be a nonempty, strictly increasing vector")
        shape = (len(self.thresholds), classes)
        self.tp = np.zeros(shape, dtype=np.int64)
        self.fp = np.zeros(shape, dtype=np.int64)
        self.fn = np.zeros(shape, dtype=np.int64)
        self.normal_fp = np.zeros(len(self.thresholds), dtype=np.int64)
        self.background = np.zeros(len(self.thresholds), dtype=np.int64)
        self.images = 0
        self.normal_images = 0

    def update(self, probabilities: np.ndarray, target: np.ndarray):
        probabilities = np.asarray(probabilities)
        target = np.asarray(target, dtype=bool)
        if probabilities.shape != target.shape or probabilities.ndim != 3 or probabilities.shape[0] != self.tp.shape[1]:
            raise ValueError("Expected matching [classes,H,W] probability and target arrays")
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Probabilities must be finite and within [0,1]")
        local_tp = np.zeros_like(self.tp)
        local_fp = np.zeros_like(self.fp)
        local_fn = np.zeros_like(self.fn)
        bins = len(self.thresholds) + 1
        for c in range(self.tp.shape[1]):
            # Bin k contains pixels above precisely k thresholds. Cumulative
            # sums count >= threshold, including exact equality at the edge.
            bin_index = np.searchsorted(self.thresholds, probabilities[c].ravel(), side="right")
            truth = target[c].ravel()
            positive = np.bincount(bin_index[truth], minlength=bins)
            negative = np.bincount(bin_index[~truth], minlength=bins)
            local_tp[:, c] = positive[::-1].cumsum()[::-1][1:]
            local_fp[:, c] = negative[::-1].cumsum()[::-1][1:]
            local_fn[:, c] = int(truth.sum()) - local_tp[:, c]
        self.tp += local_tp
        self.fp += local_fp
        self.fn += local_fn
        max_probability = float(probabilities.max())
        any_prediction = max_probability >= self.thresholds
        self.background += ~any_prediction
        self.images += 1
        is_normal = not target.any()
        if is_normal:
            self.normal_images += 1
            self.normal_fp += any_prediction
        return {"tp": local_tp, "fp": local_fp, "fn": local_fn, "is_normal": is_normal,
                "max_probability": max_probability}

    def compute(self):
        tp, fp, fn = self.tp, self.fp, self.fn
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        dice = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) != 0)
        iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fp + fn) != 0)
        return [
            {"threshold": float(threshold), "tp": tp[i].tolist(), "fp": fp[i].tolist(), "fn": fn[i].tolist(),
             "dice_per_class": dice[i].tolist(), "iou_per_class": iou[i].tolist(),
             "precision_per_class": precision[i].tolist(), "recall_per_class": recall[i].tolist(),
             "macro_dice": float(dice[i].mean()),
             "empty_image_false_positive_rate": float(self.normal_fp[i] / self.normal_images) if self.normal_images else None,
             "full_background_prediction_image_ratio": float(self.background[i] / self.images) if self.images else None,
             "normal_false_positive_images": int(self.normal_fp[i]), "normal_images": self.normal_images,
             "full_background_prediction_images": int(self.background[i]), "images": self.images}
            for i, threshold in enumerate(self.thresholds)
        ]
