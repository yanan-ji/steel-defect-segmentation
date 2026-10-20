"""Validation-only connected-component area filtering and pixel-level metrics."""
from __future__ import annotations

import cv2
import numpy as np


AREA_THRESHOLDS = (0, 8, 16, 32, 64, 128, 256)
AREA_BINS = (0, 8, 16, 32, 64, 128, 256, float("inf"))


def components(mask: np.ndarray):
    if mask.ndim != 2:
        raise ValueError("Expected a 2D mask")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(mask, dtype=np.uint8), connectivity=8
    )
    return labels, stats[:, cv2.CC_STAT_AREA].astype(np.int64), count


def filter_components(mask: np.ndarray, minimum_area: int):
    """Retain 8-connected foreground regions with area >= minimum_area."""
    if minimum_area < 0:
        raise ValueError("minimum_area must be nonnegative")
    binary = np.asarray(mask, dtype=bool)
    if minimum_area == 0:
        return binary.copy()
    labels, areas, _ = components(binary)
    return binary & (areas[labels] >= minimum_area)


class AreaSweepAccumulator:
    """Compute TP/FP/FN on filtered original-image masks for every area rule."""

    def __init__(self, thresholds=AREA_THRESHOLDS, classes=4):
        self.thresholds = tuple(int(x) for x in thresholds)
        if not self.thresholds or self.thresholds[0] != 0 or sorted(set(self.thresholds)) != list(self.thresholds):
            raise ValueError("Thresholds must be unique, sorted, and start at 0")
        shape = (len(self.thresholds), classes)
        self.tp = np.zeros(shape, np.int64)
        self.fp = np.zeros(shape, np.int64)
        self.fn = np.zeros(shape, np.int64)
        self.deleted_pixels = np.zeros(shape, np.int64)
        self.deleted_components = np.zeros(shape, np.int64)
        self.normal_fp = np.zeros(len(self.thresholds), np.int64)
        self.background = np.zeros(len(self.thresholds), np.int64)
        self.images = self.normal_images = 0
        categories = ("normal_prediction", "defect_prediction_overlapping_truth",
                      "defect_prediction_no_truth_overlap", "true_defect")
        self.component_areas = {key: [[] for _ in range(classes)] for key in categories}

    def update(self, prediction: np.ndarray, target: np.ndarray):
        prediction = np.asarray(prediction, dtype=bool)
        target = np.asarray(target, dtype=bool)
        if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[0] != self.tp.shape[1]:
            raise ValueError("Expected matching [classes,H,W] prediction and truth")
        is_normal = not target.any()
        self.images += 1
        self.normal_images += int(is_normal)
        local_fp = np.zeros_like(self.fp)
        local_fn = np.zeros_like(self.fn)
        local_removed = np.zeros_like(self.deleted_pixels)
        has_any = np.zeros(len(self.thresholds), dtype=bool)
        for c in range(self.tp.shape[1]):
            pred, truth = prediction[c], target[c]
            labels, areas, count = components(pred)
            component_areas = areas[1:count]
            if is_normal:
                self.component_areas["normal_prediction"][c].extend(component_areas.tolist())
            else:
                overlaps = np.bincount(labels[truth].ravel(), minlength=count)[1:]
                self.component_areas["defect_prediction_overlapping_truth"][c].extend(component_areas[overlaps > 0].tolist())
                self.component_areas["defect_prediction_no_truth_overlap"][c].extend(component_areas[overlaps == 0].tolist())
            if truth.any():
                _, true_areas, true_count = components(truth)
                self.component_areas["true_defect"][c].extend(true_areas[1:true_count].tolist())
            truth_count = int(np.count_nonzero(truth))
            for i, threshold in enumerate(self.thresholds):
                filtered = pred if threshold == 0 else pred & (areas[labels] >= threshold)
                # Count pixels from the full filtered original-image mask, not
                # from region counts or approximated region overlap.
                tp = int(np.count_nonzero(filtered & truth))
                fp = int(np.count_nonzero(filtered & ~truth))
                self.tp[i, c] += tp
                self.fp[i, c] += fp
                self.fn[i, c] += truth_count - tp
                local_fp[i, c] = fp
                local_fn[i, c] = truth_count - tp
                local_removed[i, c] = int(np.count_nonzero(pred)) - int(np.count_nonzero(filtered))
                self.deleted_pixels[i, c] += local_removed[i, c]
                self.deleted_components[i, c] += int(np.count_nonzero(component_areas < threshold))
                has_any[i] |= bool(filtered.any())
        self.background += ~has_any
        if is_normal:
            self.normal_fp += has_any
        return {"is_normal": is_normal, "fp": local_fp, "fn": local_fn,
                "removed_pixels": local_removed, "has_prediction": has_any}

    def compute(self):
        tp, fp, fn = self.tp, self.fp, self.fn
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
        dice = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0)
        iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fp + fn) > 0)
        return [{"minimum_area": threshold, "tp": tp[i].tolist(), "fp": fp[i].tolist(), "fn": fn[i].tolist(),
                 "dice_per_class": dice[i].tolist(), "iou_per_class": iou[i].tolist(),
                 "precision_per_class": precision[i].tolist(), "recall_per_class": recall[i].tolist(),
                 "macro_dice": float(dice[i].mean()),
                 "empty_image_false_positive_rate": float(self.normal_fp[i] / self.normal_images) if self.normal_images else None,
                 "full_background_prediction_image_ratio": float(self.background[i] / self.images) if self.images else None,
                 "normal_false_positive_images": int(self.normal_fp[i]), "normal_images": self.normal_images,
                 "full_background_prediction_images": int(self.background[i]), "images": self.images,
                 "deleted_pixels_per_class": self.deleted_pixels[i].tolist(),
                 "deleted_components_per_class": self.deleted_components[i].tolist()}
                for i, threshold in enumerate(self.thresholds)]

    def area_distributions(self):
        result = {}
        for category, classes in self.component_areas.items():
            result[category] = []
            for values in classes:
                array = np.asarray(values, dtype=np.int64)
                result[category].append({"count": len(values),
                                         "area_bins_1_7_to_256_plus": np.histogram(array, bins=AREA_BINS)[0].tolist(),
                                         "median": float(np.median(array)) if array.size else None,
                                         "p90": float(np.percentile(array, 90)) if array.size else None,
                                         "max": int(array.max()) if array.size else None})
        return result
