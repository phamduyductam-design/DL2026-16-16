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


def compute_pixel_metrics(
    *,
    masks: Sequence[object] | np.ndarray,
    anomaly_maps: Sequence[object] | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute pooled pixel-level metrics for one category."""
    mask_array = np.asarray(masks)

    try:
        score_array = np.asarray(anomaly_maps, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "anomaly_maps must contain numeric values"
        ) from exc

    if mask_array.ndim not in (2, 3):
        raise ValueError(
            "masks must be a 2D image or a 3D batch of images"
        )

    if score_array.shape != mask_array.shape:
        raise ValueError(
            "masks and anomaly_maps must have the same shape"
        )

    if mask_array.size == 0:
        raise ValueError("masks and anomaly_maps must not be empty")

    if not np.all(np.isin(mask_array, [0, 1])):
        raise ValueError("masks must contain only 0 and 1")

    if not np.all(np.isfinite(score_array)):
        raise ValueError(
            "anomaly_maps must contain only finite values"
        )

    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a finite number") from exc

    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be a finite number")

    mask_array = mask_array.astype(np.int8, copy=False)
    predicted_masks = (score_array >= threshold_value).astype(np.int8)

    flat_masks = mask_array.ravel()
    flat_predictions = predicted_masks.ravel()
    flat_scores = score_array.ravel()

    true_positive = int(
        np.sum((flat_masks == 1) & (flat_predictions == 1))
    )
    false_positive = int(
        np.sum((flat_masks == 0) & (flat_predictions == 1))
    )
    false_negative = int(
        np.sum((flat_masks == 1) & (flat_predictions == 0))
    )

    dice_denominator = (
        2 * true_positive + false_positive + false_negative
    )
    iou_denominator = (
        true_positive + false_positive + false_negative
    )

    pixel_dice = (
        2 * true_positive / dice_denominator
        if dice_denominator > 0
        else 1.0
    )
    pixel_iou = (
        true_positive / iou_denominator
        if iou_denominator > 0
        else 1.0
    )

    if np.unique(flat_masks).size < 2:
        warnings.warn(
            "pixel AUROC is undefined for single class masks",
            RuntimeWarning,
            stacklevel=2,
        )
        pixel_auroc = float("nan")
    else:
        pixel_auroc = float(
            roc_auc_score(flat_masks, flat_scores)
        )

    return {
        "pixel_auroc": pixel_auroc,
        "pixel_dice": float(pixel_dice),
        "pixel_iou": float(pixel_iou),
    }
