from __future__ import annotations

from collections.abc import Sequence

import numpy as np


_ALLOWED_QUANTILE_METHODS = {
    "higher",
    "lower",
    "nearest",
    "linear",
    "midpoint",
}


def fit_normal_threshold(
    *,
    normal_scores: Sequence[object] | np.ndarray,
    quantile: float,
    quantile_method: str = "higher",
) -> float:
    """Fit a deterministic threshold using only normal validation scores."""
    try:
        score_array = np.asarray(normal_scores, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "normal_scores must contain numeric values"
        ) from exc

    if score_array.ndim == 0:
        raise ValueError("normal_scores must be an array-like collection")

    if score_array.size == 0:
        raise ValueError("normal_scores must not be empty")

    if not np.all(np.isfinite(score_array)):
        raise ValueError(
            "normal_scores must contain only finite values"
        )

    try:
        quantile_value = float(quantile)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "quantile must be a finite number between 0 and 1"
        ) from exc

    if (
        not np.isfinite(quantile_value)
        or not 0.0 < quantile_value < 1.0
    ):
        raise ValueError(
            "quantile must be a finite number between 0 and 1"
        )

    if quantile_method not in _ALLOWED_QUANTILE_METHODS:
        raise ValueError(
            f"unsupported quantile method: {quantile_method!r}"
        )

    return float(
        np.quantile(
            score_array.ravel(),
            quantile_value,
            method=quantile_method,
        )
    )


def fit_thresholds_from_normal_validation(
    *,
    normal_image_scores: Sequence[object] | np.ndarray,
    normal_pixel_scores: Sequence[object] | np.ndarray,
    image_quantile: float = 0.99,
    pixel_quantile: float = 0.995,
    quantile_method: str = "higher",
) -> dict[str, float | str]:
    """Fit independent image and pixel thresholds from normal validation."""
    threshold_image = fit_normal_threshold(
        normal_scores=normal_image_scores,
        quantile=image_quantile,
        quantile_method=quantile_method,
    )
    threshold_pixel = fit_normal_threshold(
        normal_scores=normal_pixel_scores,
        quantile=pixel_quantile,
        quantile_method=quantile_method,
    )

    return {
        "threshold_image": threshold_image,
        "threshold_pixel": threshold_pixel,
        "threshold_source": "normal_validation",
    }
