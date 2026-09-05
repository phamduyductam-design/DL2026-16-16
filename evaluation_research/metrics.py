from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_image_metrics(
    *,
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute image-level anomaly-detection metrics.

    Labels use 0 for normal and 1 for defect. Scores greater than or
    equal to the threshold are classified as defects.
    """
    label_array = np.asarray(labels)

    try:
        score_array = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("scores must contain numeric values") from exc

    if label_array.ndim != 1:
        raise ValueError("labels must be one-dimensional")

    if score_array.ndim != 1:
        raise ValueError("scores must be one-dimensional")

    if label_array.size == 0:
        raise ValueError("labels and scores must not be empty")

    if label_array.size != score_array.size:
        raise ValueError("labels and scores must have the same length")

    if not np.all(np.isin(label_array, [0, 1])):
        raise ValueError("labels must contain only 0 and 1")

    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must contain only finite values")

    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a finite number") from exc

    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be a finite number")

    label_array = label_array.astype(np.int8, copy=False)

    if np.unique(label_array).size < 2:
        warnings.warn(
            "image metrics are undefined for single class labels",
            RuntimeWarning,
            stacklevel=2,
        )
        undefined = float("nan")
        return {
            "image_auroc": undefined,
            "image_ap": undefined,
            "image_f1": undefined,
        }

    predicted_labels = (score_array >= threshold_value).astype(np.int8)

    true_positive = int(
        np.sum((label_array == 1) & (predicted_labels == 1))
    )
    false_positive = int(
        np.sum((label_array == 0) & (predicted_labels == 1))
    )
    false_negative = int(
        np.sum((label_array == 1) & (predicted_labels == 0))
    )

    f1_denominator = (
        2 * true_positive + false_positive + false_negative
    )
    image_f1 = (
        2 * true_positive / f1_denominator
        if f1_denominator > 0
        else float("nan")
    )

    return {
        "image_auroc": float(roc_auc_score(label_array, score_array)),
        "image_ap": float(
            average_precision_score(label_array, score_array)
        ),
        "image_f1": float(image_f1),
    }
