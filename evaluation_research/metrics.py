from __future__ import annotations
from numbers import Integral, Real
import warnings
from collections.abc import Sequence
from scipy.ndimage import label as connected_components
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


def compute_aupro(
    *,
    masks: Sequence[object] | np.ndarray,
    anomaly_maps: Sequence[object] | np.ndarray,
    max_fpr: float = 0.30,
    num_thresholds: int = 200,
    connectivity: int = 8,
) -> float:
    """Compute normalized area under the per-region-overlap curve."""
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

    if isinstance(max_fpr, bool) or not isinstance(max_fpr, Real):
        raise ValueError("max_fpr must be between 0 and 1")

    max_fpr_value = float(max_fpr)

    if (
        not np.isfinite(max_fpr_value)
        or not 0.0 < max_fpr_value <= 1.0
    ):
        raise ValueError("max_fpr must be between 0 and 1")

    if (
        isinstance(num_thresholds, bool)
        or not isinstance(num_thresholds, Integral)
        or num_thresholds < 3
    ):
        raise ValueError(
            "num_thresholds must be an integer of at least 3"
        )

    if connectivity != 8:
        raise ValueError(
            "this protocol requires 8-connectivity"
        )

    if mask_array.ndim == 2:
        mask_array = mask_array[np.newaxis, ...]
        score_array = score_array[np.newaxis, ...]

    mask_array = mask_array.astype(np.uint8, copy=False)

    structure = np.ones((3, 3), dtype=np.uint8)
    regions: list[tuple[int, np.ndarray]] = []

    for image_index, mask in enumerate(mask_array):
        component_map, component_count = connected_components(
            mask,
            structure=structure,
        )

        for component_id in range(1, component_count + 1):
            regions.append(
                (image_index, component_map == component_id)
            )

    if not regions:
        warnings.warn(
            "AUPRO is undefined without a defect region",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")

    background = mask_array == 0
    background_count = int(np.count_nonzero(background))

    if background_count == 0:
        warnings.warn(
            "AUPRO is undefined without background pixels",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")

    finite_thresholds = np.linspace(
        float(np.max(score_array)),
        float(np.min(score_array)),
        int(num_thresholds),
    )
    thresholds = np.concatenate(
        (
            np.array([np.inf]),
            finite_thresholds,
        )
    )

    false_positive_rates: list[float] = []
    per_region_overlaps: list[float] = []

    for threshold in thresholds:
        prediction = score_array >= threshold

        false_positive_rate = (
            np.count_nonzero(prediction & background)
            / background_count
        )

        region_overlaps = [
            float(np.mean(prediction[image_index][region]))
            for image_index, region in regions
        ]

        false_positive_rates.append(
            float(false_positive_rate)
        )
        per_region_overlaps.append(
            float(np.mean(region_overlaps))
        )

    fpr_array = np.asarray(false_positive_rates, dtype=float)
    pro_array = np.asarray(per_region_overlaps, dtype=float)

    points_above_limit = np.flatnonzero(
        fpr_array > max_fpr_value
    )

    if points_above_limit.size:
        right_index = int(points_above_limit[0])
        left_index = right_index - 1

        left_fpr = fpr_array[left_index]
        right_fpr = fpr_array[right_index]
        left_pro = pro_array[left_index]
        right_pro = pro_array[right_index]

        interpolation_fraction = (
            (max_fpr_value - left_fpr)
            / (right_fpr - left_fpr)
        )
        boundary_pro = left_pro + interpolation_fraction * (
            right_pro - left_pro
        )

        clipped_fpr = np.concatenate(
            (
                fpr_array[:right_index],
                np.array([max_fpr_value]),
            )
        )
        clipped_pro = np.concatenate(
            (
                pro_array[:right_index],
                np.array([boundary_pro]),
            )
        )
    else:
        clipped_fpr = fpr_array
        clipped_pro = pro_array

    area = float(np.trapezoid(clipped_pro, clipped_fpr))
    normalized_area = area / max_fpr_value

    return float(np.clip(normalized_area, 0.0, 1.0))
