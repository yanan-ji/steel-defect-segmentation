import numpy as np

from steel_defect_segmentation.rle import encode_rle
from steel_defect_segmentation.sampling import build_patch_index, draw_epoch, draw_statistics, image_weights, index_statistics


def test_multilabel_patch_index_and_normal_source():
    a = np.zeros((4, 4, 8), np.uint8)
    a[0, 0, 0] = 1
    a[1, 0, 0] = 1  # same patch is genuinely multi-label
    labels = {"a": {1: encode_rle(a[0]), 2: encode_rle(a[1])}, "b": {}}
    index = build_patch_index(["a", "b"], labels, height=4, width=8, patch_width=4)
    assert index["flags"].shape == (2, 2, 4)
    assert index["flags"][0, 0].tolist() == [True, True, False, False]
    assert index["normal_source"].tolist() == [False, True]
    assert index_statistics(index)["background_patch_ratio"] == .75
    labels["validation_only"] = {3: encode_rle(np.ones((4, 8), np.uint8))}
    assert np.array_equal(index["flags"], build_patch_index(["a", "b"], labels, height=4, width=8, patch_width=4)["flags"])


def test_rare_class_weight_and_seeded_bounded_sampling():
    n = 60
    flags = np.zeros((n, 4, 4), bool)
    flags[:30, 0, 2] = True
    flags[30:45, 0, 0] = True
    flags[45:55, 0, 1] = True
    normal = np.zeros(n, bool)
    normal[55:] = True
    index = {"image_ids": [str(i) for i in range(n)], "flags": flags, "normal_source": normal}
    frequencies = [.03, .006, .23, .04]
    weights = image_weights(index, frequencies)
    assert weights[45] > weights[0]
    assert weights[30] > weights[0]
    first = draw_epoch(index, frequencies, seed=42, epoch=1, replacement_fraction=.5)
    assert np.array_equal(first, draw_epoch(index, frequencies, seed=42, epoch=1, replacement_fraction=.5))
    assert not np.array_equal(first, draw_epoch(index, frequencies, seed=42, epoch=2, replacement_fraction=.5))
    stats = draw_statistics(index, first)
    assert stats["draws"] == n
    assert stats["max_uses_of_one_image"] <= 3
    assert stats["normal_source_ratio"] == 5 / n
    assert np.bincount(flags[first].any(axis=2).sum(axis=1), minlength=5).tolist() == np.bincount(flags.any(axis=2).sum(axis=1), minlength=5).tolist()
    # Distributional property over deterministic epochs, not a guarantee for
    # every small random draw: rare class 2 rises while common class 3 falls.
    rare_counts, common_counts = [], []
    for epoch in range(1, 41):
        drawn = draw_epoch(index, frequencies, seed=42, epoch=epoch, replacement_fraction=.5)
        rare_counts.append(flags[drawn, :, 1].any(axis=1).sum())
        common_counts.append(flags[drawn, :, 2].any(axis=1).sum())
    assert np.mean(rare_counts) > 10
    assert np.mean(common_counts) < 30
