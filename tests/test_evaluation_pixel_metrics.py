import math

import pytest

from evaluation_research.metrics import compute_pixel_metrics


def test_pixel_metrics_known_overlap():
    result = compute_pixel_metrics(
        masks=[[[1, 1], [0, 0]]],
        anomaly_maps=[[[0.9, 0.4], [0.8, 0.1]]],
        threshold=0.5,
    )

    assert result["pixel_auroc"] == pytest.approx(0.75)
    assert result["pixel_dice"] == pytest.approx(0.5)
    assert result["pixel_iou"] == pytest.approx(1 / 3)


def test_pixel_metrics_micro_average_includes_normal_images():
    result = compute_pixel_metrics(
        masks=[
            [[1, 1], [0, 0]],
            [[0, 0], [0, 0]],
        ],
        anomaly_maps=[
            [[0.9, 0.4], [0.8, 0.1]],
            [[0.6, 0.1], [0.1, 0.1]],
        ],
        threshold=0.5,
    )

    assert result["pixel_auroc"] == pytest.approx(10 / 12)
    assert result["pixel_dice"] == pytest.approx(0.4)
    assert result["pixel_iou"] == pytest.approx(0.25)


def test_pixel_metrics_threshold_boundary_is_inclusive():
    with pytest.warns(RuntimeWarning, match="single class"):
        result = compute_pixel_metrics(
            masks=[[1]],
            anomaly_maps=[[0.5]],
            threshold=0.5,
        )

    assert math.isnan(result["pixel_auroc"])
    assert result["pixel_dice"] == pytest.approx(1.0)
    assert result["pixel_iou"] == pytest.approx(1.0)


def test_pixel_metrics_both_empty_returns_perfect_overlap():
    with pytest.warns(RuntimeWarning, match="single class"):
        result = compute_pixel_metrics(
            masks=[[0, 0], [0, 0]],
            anomaly_maps=[[0.1, 0.2], [0.3, 0.4]],
            threshold=0.5,
        )

    assert math.isnan(result["pixel_auroc"])
    assert result["pixel_dice"] == pytest.approx(1.0)
    assert result["pixel_iou"] == pytest.approx(1.0)


def test_pixel_metrics_false_positive_on_empty_mask_returns_zero():
    with pytest.warns(RuntimeWarning, match="single class"):
        result = compute_pixel_metrics(
            masks=[[0, 0], [0, 0]],
            anomaly_maps=[[0.5, 0.2], [0.3, 0.4]],
            threshold=0.5,
        )

    assert math.isnan(result["pixel_auroc"])
    assert result["pixel_dice"] == pytest.approx(0.0)
    assert result["pixel_iou"] == pytest.approx(0.0)


def test_pixel_metrics_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        compute_pixel_metrics(
            masks=[[0, 1], [0, 0]],
            anomaly_maps=[[0.1, 0.9]],
            threshold=0.5,
        )


@pytest.mark.parametrize(
    "invalid_score",
    [float("nan"), float("inf"), float("-inf")],
)
def test_pixel_metrics_rejects_non_finite_maps(invalid_score):
    with pytest.raises(ValueError):
        compute_pixel_metrics(
            masks=[[0, 1]],
            anomaly_maps=[[0.1, invalid_score]],
            threshold=0.5,
        )


def test_pixel_metrics_rejects_non_binary_masks():
    with pytest.raises(ValueError):
        compute_pixel_metrics(
            masks=[[0, 2]],
            anomaly_maps=[[0.1, 0.9]],
            threshold=0.5,
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [float("nan"), float("inf")],
)
def test_pixel_metrics_rejects_non_finite_threshold(invalid_threshold):
    with pytest.raises(ValueError):
        compute_pixel_metrics(
            masks=[[0, 1]],
            anomaly_maps=[[0.1, 0.9]],
            threshold=invalid_threshold,
        )


def test_pixel_metrics_rejects_one_dimensional_inputs():
    with pytest.raises(ValueError):
        compute_pixel_metrics(
            masks=[0, 1],
            anomaly_maps=[0.1, 0.9],
            threshold=0.5,
        )


def test_pixel_metrics_rejects_empty_input():
    with pytest.raises(ValueError):
        compute_pixel_metrics(
            masks=[[]],
            anomaly_maps=[[]],
            threshold=0.5,
        )
