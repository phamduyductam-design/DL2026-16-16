import csv
import json
import math

import numpy as np
import pytest

from evaluation_research.result_writer import (
    RESULT_COLUMNS,
    write_results_csv,
    write_results_json,
)


def _summary_result():
    return {
        "experiment_id": "run_001",
        "model": "patchcore",
        "category": "wood",
        "seed": 42,
        "n_test": 2,
        "image_auroc": 1.0,
        "image_ap": 1.0,
        "image_f1": 1.0,
        "pixel_auroc": 1.0,
        "pixel_aupro_30": 1.0,
        "pixel_dice": 1.0,
        "pixel_iou": 1.0,
        "threshold_image": 0.5,
        "threshold_pixel": 0.5,
        "threshold_source": "normal_validation",
        "inference_ms_mean": 15.0,
        "inference_ms_p95": 19.5,
        "config_path": "configs/run.yaml",
    }


def test_write_results_csv_creates_parent_and_preserves_columns(
    tmp_path,
):
    output_path = tmp_path / "reports" / "results.csv"

    written_path = write_results_csv(
        [_summary_result()],
        output_path,
    )

    assert written_path == output_path

    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == list(RESULT_COLUMNS)
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "run_001"
    assert rows[0]["pixel_aupro_30"] == "1.0"
    assert rows[0]["threshold_source"] == "normal_validation"


def test_write_results_json_preserves_native_values(tmp_path):
    output_path = tmp_path / "reports" / "results.json"
    result = _summary_result()

    written_path = write_results_json([result], output_path)

    assert written_path == output_path

    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded == [result]


def test_writers_preserve_result_order(tmp_path):
    first = _summary_result()
    second = _summary_result()
    second["experiment_id"] = "run_002"
    second["model"] = "padim"

    output_path = tmp_path / "results.json"
    write_results_json([first, second], output_path)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert [row["experiment_id"] for row in loaded] == [
        "run_001",
        "run_002",
    ]


def test_writers_serialize_non_finite_metrics_safely(tmp_path):
    result = _summary_result()
    result["image_auroc"] = float("nan")
    result["warnings"] = ["single class labels"]

    json_path = tmp_path / "results.json"
    csv_path = tmp_path / "results.csv"

    write_results_json([result], json_path)
    write_results_csv([result], csv_path)

    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))

    with csv_path.open(newline="", encoding="utf-8") as handle:
        loaded_csv = list(csv.DictReader(handle))

    assert loaded_json[0]["image_auroc"] is None
    assert loaded_json[0]["warnings"] == ["single class labels"]
    assert loaded_csv[0]["image_auroc"] == ""
    assert loaded_csv[0]["warnings"] == '["single class labels"]'
    assert math.isnan(result["image_auroc"])


def test_write_results_json_supports_numpy_scalars(tmp_path):
    result = _summary_result()
    result["seed"] = np.int64(42)
    result["image_auroc"] = np.float64(0.75)

    output_path = tmp_path / "results.json"
    write_results_json([result], output_path)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded[0]["seed"] == 42
    assert loaded[0]["image_auroc"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("writer", "suffix"),
    [
        (write_results_csv, ".csv"),
        (write_results_json, ".json"),
    ],
)
def test_writers_refuse_to_overwrite_existing_files(
    tmp_path,
    writer,
    suffix,
):
    output_path = tmp_path / f"results{suffix}"
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        writer([_summary_result()], output_path)

    assert output_path.read_text(encoding="utf-8") == "existing"


@pytest.mark.parametrize(
    ("writer", "suffix"),
    [
        (write_results_csv, ".csv"),
        (write_results_json, ".json"),
    ],
)
def test_writers_reject_empty_results(tmp_path, writer, suffix):
    with pytest.raises(ValueError):
        writer([], tmp_path / f"results{suffix}")


@pytest.mark.parametrize(
    ("writer", "suffix"),
    [
        (write_results_csv, ".csv"),
        (write_results_json, ".json"),
    ],
)
def test_writers_reject_missing_required_fields(
    tmp_path,
    writer,
    suffix,
):
    result = _summary_result()
    result.pop("pixel_aupro_30")

    with pytest.raises(ValueError):
        writer([result], tmp_path / f"results{suffix}")


def test_csv_writer_requires_csv_suffix(tmp_path):
    with pytest.raises(ValueError):
        write_results_csv(
            [_summary_result()],
            tmp_path / "results.json",
        )


def test_json_writer_requires_json_suffix(tmp_path):
    with pytest.raises(ValueError):
        write_results_json(
            [_summary_result()],
            tmp_path / "results.csv",
        )
