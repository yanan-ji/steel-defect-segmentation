import numpy as np
import pytest

from steel_defect_segmentation.thresholds import ThresholdAccumulator, THRESHOLDS


def test_grid_includes_original_half_threshold():
    assert len(THRESHOLDS) == 18
    assert THRESHOLDS[0] == pytest.approx(.10)
    assert THRESHOLDS[-1] == pytest.approx(.95)
    assert .5 in THRESHOLDS


def test_streaming_matches_brute_force_including_equal_threshold_and_empty_image():
    thresholds = np.array([.25, .5, .75])
    acc = ThresholdAccumulator(thresholds)
    p1 = np.array([[[.25, .49]], [[.5, .75]], [[.1, .9]], [[.2, .3]]])
    t1 = np.array([[[1, 0]], [[0, 1]], [[0, 0]], [[0, 0]]])
    p2 = np.array([[[.1, .2]], [[.1, .2]], [[.1, .2]], [[.1, .2]]])
    t2 = np.zeros_like(t1)
    acc.update(p1, t1)
    acc.update(p2, t2)
    rows = acc.compute()
    for i, threshold in enumerate(thresholds):
        pred = np.stack([p1 >= threshold, p2 >= threshold])
        truth = np.stack([t1, t2]).astype(bool)
        tp = (pred & truth).sum((0, 2, 3)).tolist()
        fp = (pred & ~truth).sum((0, 2, 3)).tolist()
        fn = (~pred & truth).sum((0, 2, 3)).tolist()
        assert rows[i]["tp"] == tp
        assert rows[i]["fp"] == fp
        assert rows[i]["fn"] == fn
        assert rows[i]["normal_images"] == 1
        assert rows[i]["normal_false_positive_images"] == int((p2 >= threshold).any())
        assert rows[i]["full_background_prediction_images"] == int(not (p1 >= threshold).any()) + int(not (p2 >= threshold).any())


def test_rejects_wrong_shape_and_nonfinite_probability():
    acc = ThresholdAccumulator()
    with pytest.raises(ValueError):
        acc.update(np.zeros((4, 2, 2)), np.zeros((3, 2, 2)))
    p = np.zeros((4, 2, 2))
    p[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        acc.update(p, np.zeros_like(p))
