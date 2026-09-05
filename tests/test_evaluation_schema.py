import numpy as np
import pytest

from evaluation_research.schema import validate_prediction


@pytest.fixture
def valid_prediction(tmp_path):
    artifact_root = tmp_path / "artifacts"
    output_dir = artifact_root / "outputs" / "run_001"
    config_dir = artifact_root / "configs"

    output_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    np.save(
        output_dir / "anomaly_map.npy",
        np.array([[0.1, 0.2], [0.3, 0.9]], dtype=float),
    )
    (output_dir / "mask.png").write_bytes(b"mask fixture")
    (output_dir / "overlay.png").write_bytes(b"overlay fixture")
    (config_dir / "run.yaml").write_text(
        "experiment_id: run_001\n",
        encoding="utf-8",
    )

    record = {
        "experiment_id": "run_001",
        "model": "patchcore",
        "category": "wood",
        "sample_id": "000123",
        "anomaly_score": 0.9,
        "threshold_image": 0.5,
        "predicted_label": "defect",
        "anomaly_map_path": "outputs/run_001/anomaly_map.npy",
        "binary_mask_path": "outputs/run_001/mask.png",
        "overlay_path": "outputs/run_001/overlay.png",
        "inference_ms": 18.4,
        "seed": 42,
        "config_path": "configs/run.yaml",
    }

    return record, artifact_root


def test_validate_prediction_accepts_valid_record(valid_prediction):
    record, artifact_root = valid_prediction

    validated = validate_prediction(
        record,
        artifact_root=artifact_root,
        expected_map_shape=(2, 2),
    )

    assert validated == record


@pytest.mark.parametrize(
    "missing_field",
    [
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
    ],
)
def test_validate_prediction_rejects_missing_fields(
    valid_prediction,
    missing_field,
):
    record, artifact_root = valid_prediction
    record.pop(missing_field)

    with pytest.raises(ValueError):
        validate_prediction(record, artifact_root=artifact_root)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("experiment_id", ""),
        ("model", ""),
        ("category", ""),
        ("sample_id", ""),
    ],
)
def test_validate_prediction_rejects_empty_identifiers(
    valid_prediction,
    field,
    invalid_value,
):
    record, artifact_root = valid_prediction
    record[field] = invalid_value

    with pytest.raises(ValueError):
        validate_prediction(record, artifact_root=artifact_root)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("anomaly_score", float("nan")),
        ("anomaly_score", float("inf")),
        ("threshold_image", float("nan")),
        ("inference_ms", float("inf")),
        ("inference_ms", -1.0),
    ],
)
def test_validate_prediction_rejects_invalid_numbers(
    valid_prediction,
    field,
    invalid_value,
):
    record, artifact_root = valid_prediction
    record[field] = invalid_value

    with pytest.raises(ValueError):
        validate_prediction(record, artifact_root=artifact_root)


def test_validate_prediction_rejects_invalid_label(valid_prediction):
    record, artifact_root = valid_prediction
    record["predicted_label"] = "unknown"

    with pytest.raises(ValueError):
        validate_prediction(record, artifact_root=artifact_root)


def test_validate_prediction_rejects_inconsistent_label(valid_prediction):
    record, artifact_root = valid_prediction
    record["predicted_label"] = "normal"

    with pytest.raises(ValueError):
        validate_prediction(record, artifact_root=artifact_root)


def test_validate_prediction_rejects_non_integer_seed(valid_prediction):
    record, artifact_root = valid_prediction
    record["seed"] = 42.5

    with pytest.raises(ValueError):
        validate_prediction(record, artifact_root=artifact_root)


@pytest.mark.parametrize(
    "path_field",
    [
        "anomaly_map_path",
        "binary_mask_path",
        "overlay_path",
        "config_path",
    ],
)
def test_validate_prediction_rejects_absolute_paths(
    valid_prediction,
    path_field,
):
    record, artifact_root = valid_prediction
    record[path_field] = str(
        artifact_root / record[path_field]
    )

    with pytest.raises(ValueError):
        validate_prediction(record, artifact_root=artifact_root)


def test_validate_prediction_rejects_path_traversal(valid_prediction):
    record, artifact_root = valid_prediction
    outside_path = artifact_root.parent / "outside.npy"
    np.save(outside_path, np.zeros((2, 2)))
    record["anomaly_map_path"] = "../outside.npy"

    with pytest.raises(ValueError):
        validate_prediction(record, artifact_root=artifact_root)


def test_validate_prediction_rejects_missing_artifact(valid_prediction):
    record, artifact_root = valid_prediction
    record["overlay_path"] = "outputs/run_001/missing.png"

    with pytest.raises(ValueError):
        validate_prediction(record, artifact_root=artifact_root)


def test_validate_prediction_rejects_non_2d_anomaly_map(
    valid_prediction,
):
    record, artifact_root = valid_prediction
    np.save(
        artifact_root / record["anomaly_map_path"],
        np.array([0.1, 0.2]),
    )

    with pytest.raises(ValueError):
        validate_prediction(record, artifact_root=artifact_root)


def test_validate_prediction_rejects_non_finite_anomaly_map(
    valid_prediction,
):
    record, artifact_root = valid_prediction
    np.save(
        artifact_root / record["anomaly_map_path"],
        np.array([[0.1, float("nan")], [0.2, 0.3]]),
    )

    with pytest.raises(ValueError):
        validate_prediction(record, artifact_root=artifact_root)


def test_validate_prediction_rejects_unexpected_map_shape(
    valid_prediction,
):
    record, artifact_root = valid_prediction

    with pytest.raises(ValueError):
        validate_prediction(
            record,
            artifact_root=artifact_root,
           expected_map_shape=(3, 3),
        )
