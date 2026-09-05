from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np


REQUIRED_PREDICTION_FIELDS = frozenset(
    {
        "experiment_id",
        "model",
        "category",
        "sample_id",
        "anomaly_score",
        "threshold_image",
        "predicted_label",
        "anomaly_map_path",
        "binary_mask_path",
        "overlay_path",
        "inference_ms",
        "seed",
        "config_path",
    }
)

_IDENTIFIER_FIELDS = (
    "experiment_id",
    "model",
    "category",
    "sample_id",
)

_PATH_FIELDS = (
    "anomaly_map_path",
    "binary_mask_path",
    "overlay_path",
    "config_path",
)


def _require_finite_number(
    record: Mapping[str, object],
    field: str,
) -> float:
    value = record[field]

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")

    number = float(value)

    if not np.isfinite(number):
        raise ValueError(f"{field} must be a finite number")

    return number


def _resolve_artifact_path(
    *,
    artifact_root: Path,
    relative_path: object,
    field: str,
) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"{field} must be a non-empty relative path")

    path = Path(relative_path)

    if path.is_absolute():
        raise ValueError(f"{field} must be a relative path")

    if ".." in path.parts:
        raise ValueError(f"{field} must not contain path traversal")

    resolved = (artifact_root / path).resolve()

    if not resolved.is_relative_to(artifact_root):
        raise ValueError(f"{field} resolves outside artifact_root")

    if not resolved.is_file():
        raise ValueError(f"{field} does not exist: {relative_path}")

    return resolved


def validate_prediction(
    record: Mapping[str, object],
    *,
    artifact_root: str | Path,
    expected_map_shape: tuple[int, int] | None = None,
) -> dict[str, object]:
    """Validate one prediction against the shared output contract."""
    if not isinstance(record, Mapping):
        raise ValueError("prediction record must be a mapping")

    missing_fields = REQUIRED_PREDICTION_FIELDS.difference(record)

    if missing_fields:
        fields = ", ".join(sorted(missing_fields))
        raise ValueError(f"prediction is missing required fields: {fields}")

    for field in _IDENTIFIER_FIELDS:
        value = record[field]

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")

    anomaly_score = _require_finite_number(record, "anomaly_score")
    threshold_image = _require_finite_number(record, "threshold_image")
    inference_ms = _require_finite_number(record, "inference_ms")

    if inference_ms < 0:
        raise ValueError("inference_ms must be non-negative")

    seed = record["seed"]

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")

    predicted_label = record["predicted_label"]

    if predicted_label not in {"normal", "defect"}:
        raise ValueError(
            "predicted_label must be either 'normal' or 'defect'"
        )

    expected_label = (
        "defect"
        if anomaly_score >= threshold_image
        else "normal"
    )

    if predicted_label != expected_label:
        raise ValueError(
            "predicted_label is inconsistent with anomaly_score "
            "and threshold_image"
        )

    root = Path(artifact_root).resolve()

    if not root.is_dir():
        raise ValueError("artifact_root must be an existing directory")

    resolved_paths = {
        field: _resolve_artifact_path(
            artifact_root=root,
            relative_path=record[field],
            field=field,
        )
        for field in _PATH_FIELDS
    }

    anomaly_map_path = resolved_paths["anomaly_map_path"]

    try:
        anomaly_map = np.load(anomaly_map_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            "anomaly_map_path must reference a valid NumPy array"
        ) from exc

    if anomaly_map.ndim != 2:
        raise ValueError("anomaly map must be a 2D array")

    if anomaly_map.size == 0:
        raise ValueError("anomaly map must not be empty")

    if not np.issubdtype(anomaly_map.dtype, np.number):
        raise ValueError("anomaly map must contain numeric values")

    if not np.all(np.isfinite(anomaly_map)):
        raise ValueError(
            "anomaly map must contain only finite values"
        )

    if (
        expected_map_shape is not None
        and tuple(anomaly_map.shape) != tuple(expected_map_shape)
    ):
        raise ValueError(
            "anomaly map shape does not match expected evaluation shape"
        )

    return dict(record)
