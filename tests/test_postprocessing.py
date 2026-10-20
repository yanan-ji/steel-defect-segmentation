import numpy as np
import pytest

from steel_defect_segmentation.postprocessing import AreaSweepAccumulator, filter_components


def test_eight_connected_filter_keeps_equal_area_and_diagonal():
    mask = np.zeros((6, 6), bool)
    mask[0, 0] = mask[1, 1] = True  # one 8-connected area-2 region
    mask[5, 5] = True
    assert filter_components(mask, 0).sum() == 3
    assert filter_components(mask, 2).sum() == 2
    assert filter_components(mask, 3).sum() == 0
    with pytest.raises(ValueError):
        filter_components(mask, -1)


def test_pixel_confusion_and_normal_image_false_positive():
    acc = AreaSweepAccumulator((0, 2, 3))
    pred = np.zeros((4, 6, 6), bool)
    truth = np.zeros_like(pred)
    pred[0, 0, 0] = pred[0, 1, 1] = True
    pred[0, 5, 5] = True
    truth[0, 0, 0] = truth[0, 1, 1] = True
    acc.update(pred, truth)
    normal_pred = np.zeros_like(pred)
    normal_pred[1, 2, 2] = True
    acc.update(normal_pred, np.zeros_like(pred))
    rows = acc.compute()
    assert [r["tp"][0] for r in rows] == [2, 2, 0]
    assert [r["fp"][0] for r in rows] == [1, 0, 0]
    assert [r["fn"][0] for r in rows] == [0, 0, 2]
    assert [r["normal_false_positive_images"] for r in rows] == [1, 0, 0]
    assert rows[1]["deleted_pixels_per_class"][0] == 1
    assert rows[1]["deleted_components_per_class"][0] == 1
    assert acc.area_distributions()["true_defect"][0]["count"] == 1
