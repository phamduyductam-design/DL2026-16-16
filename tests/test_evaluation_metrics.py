import math

import pytest

from evaluation_research.metrics import compute_image_metrics


def test_image_metrics_perfect_predictions():
    result = compute_image_metrics(
        labels=[0, 0, 1, 1],
        scores=[0.1, 0.2, 0.8, 0.9],
        threshold=0.5,
    )

    assert result["image_auroc"] == pytest.approx(1.0)
    assert result["image_ap"] == pytest.approx(1.0)
    assert result["image_f1"] == pytest.approx(1.0)


def test_image_metrics_threshold_boundary_is_inclusive():
    result = compute_image_metrics(
        labels=[0, 1],
        scores=[0.3, 0.5],
        threshold=0.5,
    )

    assert result["image_f1"] == pytest.approx(1.0)


def test_image_metrics_imperfect_separation():
    result = compute_image_metrics(
        labels=[0, 1, 0, 1],
        scores=[0.4, 0.3, 0.6, 0.9],
        threshold=0.5,
    )

    assert result["image_auroc"] == pytest.approx(0.5)
    assert result["image_ap"] == pytest.approx(0.75)
    assert result["image_f1"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        ([0, 0, 0], [0.1, 0.2, 0.3]),
        ([1, 1, 1], [0.6, 0.7, 0.8]),
    ],
)
def test_image_metrics_single_class_returns_nan_with_warning(labels, scores):
    with pytest.warns(RuntimeWarning, match="single class"):
        result = compute_image_metrics(
            labels=labels,
            scores=scores,
            threshold=0.5,
        )

    assert math.isnan(result["image_auroc"])
    assert math.isnan(result["image_ap"])
    assert math.isnan(result["image_f1"])


def test_image_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_image_metrics(
            labels=[0, 1],
            scores=[0.1],
            threshold=0.5,
        )


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), float("-inf")])
def test_image_metrics_rejects_non_finite_scores(invalid_score):
    with pytest.raises(ValueError):
        compute_image_metrics(
            labels=[0, 1],
            scores=[0.1, invalid_score],
            threshold=0.5,
        )


@pytest.mark.parametrize("invalid_threshold", [float("nan"), float("inf")])
def test_image_metrics_rejects_non_finite_threshold(invalid_threshold):
    with pytest.raises(ValueError):
        compute_image_metrics(
            labels=[0, 1],
            scores=[0.1, 0.9],
            threshold=invalid_threshold,
        )


def test_image_metrics_rejects_non_binary_labels():
    with pytest.raises(ValueError):
        compute_image_metrics(
            labels=[0, 2],
            scores=[0.1, 0.9],
            threshold=0.5,
        )


def test_image_metrics_rejects_empty_input():
    with pytest.raises(ValueError):
        compute_image_metrics(
            labels=[],
            scores=[],
            threshold=0.5,
        )
