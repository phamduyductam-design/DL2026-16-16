import numpy as np

from evaluation.metrics import dice_score, intersection_over_union
from evaluation.thresholds import percentile_threshold


def test_perfect_masks_score_one():
    mask = np.array([[0, 1], [1, 0]], dtype=bool)
    assert dice_score(mask, mask) == 1.0
    assert intersection_over_union(mask, mask) == 1.0


def test_threshold_uses_normal_validation_scores():
    threshold = percentile_threshold(np.array([0.1, 0.2, 0.3, 0.4]), percentile=75)
    assert 0.3 < threshold < 0.4
