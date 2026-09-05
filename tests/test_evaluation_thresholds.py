import pytest

from evaluation_research.thresholds import (
    fit_normal_threshold,
    fit_thresholds_from_normal_validation,
)


def test_normal_threshold_uses_higher_quantile():
    threshold = fit_normal_threshold(
        normal_scores=[0.1, 0.2, 0.3, 0.4],
        quantile=0.75,
        quantile_method="higher",
    )

    assert threshold == pytest.approx(0.4)


def test_normal_threshold_flattens_pixel_maps():
    threshold = fit_normal_threshold(
        normal_scores=[
            [[0.01, 0.02], [0.03, 0.04]],
        ],
        quantile=0.5,
        quantile_method="higher",
    )

    assert threshold == pytest.approx(0.03)


def test_image_and_pixel_thresholds_are_fitted_independently():
    result = fit_thresholds_from_normal_validation(
        normal_image_scores=[0.1, 0.2, 0.3, 0.4],
        normal_pixel_scores=[
            [[0.01, 0.02], [0.03, 0.04]],
        ],
        image_quantile=0.75,
        pixel_quantile=0.5,
    )

    assert result["threshold_image"] == pytest.approx(0.4)
    assert result["threshold_pixel"] == pytest.approx(0.03)
    assert result["threshold_source"] == "normal_validation"


def test_normal_threshold_is_deterministic():
    kwargs = {
        "normal_scores": [0.4, 0.1, 0.3, 0.2],
        "quantile": 0.75,
        "quantile_method": "higher",
    }

    first = fit_normal_threshold(**kwargs)
    second = fit_normal_threshold(**kwargs)

    assert first == second


def test_normal_threshold_rejects_empty_scores():
    with pytest.raises(ValueError):
        fit_normal_threshold(
            normal_scores=[],
            quantile=0.99,
        )


@pytest.mark.parametrize(
    "invalid_score",
    [float("nan"), float("inf"), float("-inf")],
)
def test_normal_threshold_rejects_non_finite_scores(invalid_score):
    with pytest.raises(ValueError):
        fit_normal_threshold(
            normal_scores=[0.1, invalid_score],
            quantile=0.99,
        )


@pytest.mark.parametrize(
    "invalid_quantile",
    [-0.1, 0.0, 1.0, 1.1, float("nan")],
)
def test_normal_threshold_rejects_invalid_quantile(invalid_quantile):
    with pytest.raises(ValueError):
        fit_normal_threshold(
            normal_scores=[0.1, 0.2],
            quantile=invalid_quantile,
        )


def test_normal_threshold_rejects_unknown_quantile_method():
    with pytest.raises(ValueError):
        fit_normal_threshold(
            normal_scores=[0.1, 0.2],
            quantile=0.99,
            quantile_method="unknown",
        )
