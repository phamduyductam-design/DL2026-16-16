import math

import pytest

from evaluation_research.metrics import compute_aupro


def test_aupro_perfect_localization_is_one():
    result = compute_aupro(
        masks=[[1, 0], [0, 0]],
        anomaly_maps=[[0.9, 0.1], [0.1, 0.1]],
    )

    assert result == pytest.approx(1.0)


def test_aupro_inverted_ranking_is_zero():
    result = compute_aupro(
        masks=[[1, 0], [0, 0]],
        anomaly_maps=[[0.1, 0.9], [0.9, 0.9]],
    )

    assert result == pytest.approx(0.0)


def test_aupro_uses_eight_connectivity():
    result = compute_aupro(
        masks=[
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        anomaly_maps=[
            [0.9, 0.9, 0.6],
            [0.6, 0.6, 0.1],
            [0.6, 0.6, 0.6],
        ],
    )

    assert result == pytest.approx(2 / 3)


def test_aupro_weights_regions_equally_not_by_pixel_count():
    result = compute_aupro(
        masks=[
            [1, 0, 0, 1],
            [0, 0, 1, 1],
        ],
        anomaly_maps=[
            [0.9, 0.8, 0.8, 0.1],
            [0.0, 0.0, 0.1, 0.1],
        ],
    )

    assert result == pytest.approx(0.5)


def test_aupro_fpr_includes_background_from_normal_images():
    result = compute_aupro(
        masks=[
            [[1, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ],
        anomaly_maps=[
            [[0.8, 0.0], [0.0, 0.0]],
            [[0.9, 0.0], [0.0, 0.0]],
        ],
    )

    assert result == pytest.approx(11 / 21)


def test_aupro_interpolates_at_max_fpr():
    result = compute_aupro(
        masks=[
            [1, 0, 0],
            [0, 1, 0],
        ],
        anomaly_maps=[
            [0.9, 0.8, 0.7],
            [0.1, 0.7, 0.1],
        ],
    )

    assert result == pytest.approx(61 / 120)


def test_aupro_without_defect_regions_returns_nan():
    with pytest.warns(RuntimeWarning, match="defect region"):
        result = compute_aupro(
            masks=[[0, 0], [0, 0]],
            anomaly_maps=[[0.1, 0.2], [0.3, 0.4]],
        )

    assert math.isnan(result)


def test_aupro_without_background_returns_nan():
    with pytest.warns(RuntimeWarning, match="background"):
        result = compute_aupro(
            masks=[[1, 1], [1, 1]],
            anomaly_maps=[[0.1, 0.2], [0.3, 0.4]],
        )

    assert math.isnan(result)


def test_aupro_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        compute_aupro(
            masks=[[1, 0], [0, 0]],
            anomaly_maps=[[0.9, 0.1]],
        )


def test_aupro_rejects_non_binary_masks():
    with pytest.raises(ValueError):
        compute_aupro(
            masks=[[1, 2], [0, 0]],
            anomaly_maps=[[0.9, 0.1], [0.1, 0.1]],
        )


@pytest.mark.parametrize(
    "invalid_score",
    [float("nan"), float("inf"), float("-inf")],
)
def test_aupro_rejects_non_finite_maps(invalid_score):
    with pytest.raises(ValueError):
        compute_aupro(
            masks=[[1, 0]],
            anomaly_maps=[[0.9, invalid_score]],
        )


@pytest.mark.parametrize(
    "invalid_max_fpr",
    [0.0, -0.1, 1.1, float("nan"), float("inf")],
)
def test_aupro_rejects_invalid_max_fpr(invalid_max_fpr):
    with pytest.raises(ValueError):
        compute_aupro(
            masks=[[1, 0]],
            anomaly_maps=[[0.9, 0.1]],
            max_fpr=invalid_max_fpr,
        )


@pytest.mark.parametrize(
    "invalid_count",
    [2, 3.5, True],
)
def test_aupro_rejects_invalid_threshold_count(invalid_count):
    with pytest.raises(ValueError):
        compute_aupro(
            masks=[[1, 0]],
            anomaly_maps=[[0.9, 0.1]],
            num_thresholds=invalid_count,
        )


@pytest.mark.parametrize("invalid_connectivity", [4, 6])
def test_aupro_rejects_non_protocol_connectivity(
    invalid_connectivity,
):
    with pytest.raises(ValueError):
        compute_aupro(
            masks=[[1, 0]],
            anomaly_maps=[[0.9, 0.1]],
            connectivity=invalid_connectivity,
        )


def test_aupro_rejects_one_dimensional_input():
    with pytest.raises(ValueError):
        compute_aupro(
            masks=[1, 0],
            anomaly_maps=[0.9, 0.1],
        )


def test_aupro_rejects_empty_input():
    with pytest.raises(ValueError):
        compute_aupro(
            masks=[[]],
            anomaly_maps=[[]],
        )
