from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


RESULT_COLUMNS = (
    "experiment_id",
    "model",
    "category",
    "seed",
    "n_test",
    "image_auroc",
    "image_ap",
    "image_f1",
    "pixel_auroc",
    "pixel_aupro_30",
    "pixel_dice",
    "pixel_iou",
    "threshold_image",
    "threshold_pixel",
    "threshold_source",
    "inference_ms_mean",
    "inference_ms_p95",
    "config_path",
)


def _normalize_value(value: object) -> object:
    """Convert values into strict JSON-compatible Python objects."""
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, np.ndarray):
        return [_normalize_value(item) for item in value.tolist()]

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _normalize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]

    if value is None or isinstance(value, (str, int, bool)):
        return value

    raise ValueError(
        f"unsupported result value type: {type(value).__name__}"
    )


def _prepare_results(
    results: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    records = list(results)

    if not records:
        raise ValueError("results must not be empty")

    required_fields = set(RESULT_COLUMNS)
    normalized_records: list[dict[str, object]] = []
    extra_fields: set[str] = set()

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"result at index {index} must be a mapping"
            )

        if not all(isinstance(key, str) for key in record):
            raise ValueError("all result field names must be strings")

        missing_fields = required_fields.difference(record)

        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"result at index {index} is missing fields: {missing}"
            )

        normalized = {
            key: _normalize_value(value)
            for key, value in record.items()
        }
        normalized_records.append(normalized)
        extra_fields.update(set(normalized).difference(required_fields))

    columns = list(RESULT_COLUMNS) + sorted(extra_fields)

    return normalized_records, columns


def _validate_output_path(
    output_path: str | Path,
    *,
    expected_suffix: str,
) -> Path:
    path = Path(output_path)

    if path.suffix.lower() != expected_suffix:
        raise ValueError(
            f"output path must use the {expected_suffix} suffix"
        )

    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing result file: {path}"
        )

    return path


def _csv_value(value: object) -> object:
    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )

    return value


def write_results_csv(
    results: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    """Write flat evaluation summaries as a UTF-8 CSV file."""
    path = _validate_output_path(
        output_path,
        expected_suffix=".csv",
    )
    records, columns = _prepare_results(results)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    column: _csv_value(record.get(column))
                    for column in columns
                }
            )

    return path


def write_results_json(
    results: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    """Write evaluation summaries as strict UTF-8 JSON."""
    path = _validate_output_path(
        output_path,
        expected_suffix=".json",
    )
    records, _ = _prepare_results(results)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            records,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")

    return path
