from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path

import numpy as np

from evaluation_research.metrics import (
    compute_image_metrics,
    compute_pixel_metrics,
)
from evaluation_research.schema import validate_prediction


_GROUP_FIELDS = (
    "experiment_id",
    "model",
    "category",
    "seed",
    "config_path",
    "threshold_image",
)


def _validate_pixel_threshold(threshold_pixel: object) -> float:
    if (
        isinstance(threshold_pixel, bool)
        or not isinstance(threshold_pixel, Real)
    ):
        raise ValueError("threshold_pixel must be a finite number")

    threshold = float(threshold_pixel)

    if not np.isfinite(threshold):
        raise ValueError("threshold_pixel must be a finite number")

    return threshold


def _validate_ground_truth_mask(
    mask: object,
    *,
    sample_id: str,
) -> np.ndarray:
    mask_array = np.asarray(mask)

    if mask_array.ndim != 2:
        raise ValueError(
            f"ground-truth mask for {sample_id!r} must be 2D"
        )

    if mask_array.size == 0:
        raise ValueError(
            f"ground-truth mask for {sample_id!r} must not be empty"
        )

    if not np.all(np.isin(mask_array, [0, 1])):
        raise ValueError(
            f"ground-truth mask for {sample_id!r} must be binary"
        )

    return mask_array.astype(np.uint8, copy=False)


def evaluate_predictions(
    predictions: Sequence[Mapping[str, object]],
    *,
    image_labels: Mapping[str, int],
    ground_truth_masks: Mapping[str, object],
    artifact_root: str | Path,
    threshold_pixel: float,
    threshold_source: str = "normal_validation",
) -> dict[str, object]:
    """Validate and evaluate predictions from one experiment run."""
    if not predictions:
        raise ValueError("predictions must not be empty")

    if threshold_source != "normal_validation":
        raise ValueError(
            "main evaluation threshold_source must be normal_validation"
        )

    pixel_threshold = _validate_pixel_threshold(threshold_pixel)
    root = Path(artifact_root).resolve()

    validated_records: list[dict[str, object]] = []
    labels: list[int] = []
    scores: list[float] = []
    masks: list[np.ndarray] = []
    anomaly_maps: list[np.ndarray] = []
    inference_times: list[float] = []
    seen_sample_ids: set[str] = set()

    for record in predictions:
        if not isinstance(record, Mapping):
            raise ValueError("each prediction must be a mapping")

        sample_id_value = record.get("sample_id")

        if (
            not isinstance(sample_id_value, str)
            or not sample_id_value.strip()
        ):
            validate_prediction(record, artifact_root=root)
            raise ValueError("sample_id must be a non-empty string")

        sample_id = sample_id_value

        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")

        if sample_id not in image_labels:
            raise ValueError(
                f"missing image label for sample_id: {sample_id}"
            )

        if sample_id not in ground_truth_masks:
            raise ValueError(
                f"missing ground-truth mask for sample_id: {sample_id}"
            )

        ground_truth_mask = _validate_ground_truth_mask(
            ground_truth_masks[sample_id],
            sample_id=sample_id,
        )

        validated = validate_prediction(
            record,
            artifact_root=root,
            expected_map_shape=tuple(ground_truth_mask.shape),
        )

        anomaly_map_path = (
            root / str(validated["anomaly_map_path"])
        ).resolve()
        anomaly_map = np.load(
            anomaly_map_path,
            allow_pickle=False,
        ).astype(float, copy=False)

        seen_sample_ids.add(sample_id)
        validated_records.append(validated)
        labels.append(image_labels[sample_id])
        scores.append(float(validated["anomaly_score"]))
        masks.append(ground_truth_mask)
        anomaly_maps.append(anomaly_map)
        inference_times.append(float(validated["inference_ms"]))

    reference = validated_records[0]

    for record in validated_records[1:]:
        for field in _GROUP_FIELDS:
            if record[field] != reference[field]:
                raise ValueError(
                    f"predictions contain mixed values for {field}"
                )

    try:
        stacked_masks = np.stack(masks)
        stacked_anomaly_maps = np.stack(anomaly_maps)
    except ValueError as exc:
        raise ValueError(
            "all masks and anomaly maps must share the same shape"
        ) from exc

    image_metrics = compute_image_metrics(
        labels=labels,
        scores=scores,
        threshold=float(reference["threshold_image"]),
    )
    pixel_metrics = compute_pixel_metrics(
        masks=stacked_masks,
        anomaly_maps=stacked_anomaly_maps,
        threshold=pixel_threshold,
    )

    inference_array = np.asarray(inference_times, dtype=float)

    return {
        "experiment_id": reference["experiment_id"],
        "model": reference["model"],
        "category": reference["category"],
        "seed": reference["seed"],
        "n_test": len(validated_records),
        "config_path": reference["config_path"],
        **image_metrics,
        **pixel_metrics,
        "threshold_image": float(reference["threshold_image"]),
        "threshold_pixel": pixel_threshold,
        "threshold_source": threshold_source,
        "inference_ms_mean": float(np.mean(inference_array)),
        "inference_ms_p95": float(
            np.percentile(inference_array, 95)
        ),
    }
