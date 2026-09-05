import numpy as np
import pytest

from evaluation_research.evaluator import evaluate_predictions


def _write_prediction(
    artifact_root,
    *,
    sample_id,
    anomaly_score,
    anomaly_map,
    inference_ms,
):
    output_dir = artifact_root / "outputs" / sample_id
    output_dir.mkdir(parents=True)

    anomaly_map_path = output_dir / "anomaly_map.npy"
    np.save(anomaly_map_path, np.asarray(anomaly_map, dtype=float))

    mask_path = output_dir / "mask.png"
    overlay_path = output_dir / "overlay.png"
    mask_path.write_bytes(b"mask fixture")
    overlay_path.write_bytes(b"overlay fixture")

    predicted_label = (
        "defect" if anomaly_score >= 0.5 else "normal"
    )

    return {
        "experiment_id": "run_001",
        "model": "patchcore",
        "category": "wood",
        "sample_id": sample_id,
        "anomaly_score": anomaly_score,
        "threshold_image": 0.5,
        "predicted_label": predicted_label,
        "anomaly_map_path": str(
            anomaly_map_path.relative_to(artifact_root)
        ),
        "binary_mask_path": str(mask_path.relative_to(artifact_root)),
        "overlay_path": str(overlay_path.relative_to(artifact_root)),
        "inference_ms": inference_ms,
        "seed": 42,
        "config_path": "configs/run.yaml",
    }


@pytest.fixture
def evaluator_case(tmp_path):
    artifact_root = tmp_path / "artifacts"
    config_dir = artifact_root / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "run.yaml").write_text(
        "experiment_id: run_001\n",
        encoding="utf-8",
    )

    predictions = [
        _write_prediction(
            artifact_root,
            sample_id="normal_001",
            anomaly_score=0.1,
            anomaly_map=[[0.1, 0.1], [0.1, 0.1]],
            inference_ms=10.0,
        ),
        _write_prediction(
            artifact_root,
            sample_id="defect_001",
            anomaly_score=0.9,
            anomaly_map=[[0.9, 0.1], [0.1, 0.1]],
            inference_ms=20.0,
        ),
    ]

    image_labels = {
        "normal_001": 0,
        "defect_001": 1,
    }
    ground_truth_masks = {
        "normal_001": np.zeros((2, 2), dtype=np.uint8),
        "defect_001": np.array(
            [[1, 0], [0, 0]],
            dtype=np.uint8,
        ),
    }

    return {
        "artifact_root": artifact_root,
        "predictions": predictions,
        "image_labels": image_labels,
        "ground_truth_masks": ground_truth_masks,
    }


def test_evaluate_predictions_returns_flat_summary(evaluator_case):
    result = evaluate_predictions(
        **evaluator_case,
        threshold_pixel=0.5,
        threshold_source="normal_validation",
    )

    assert result["experiment_id"] == "run_001"
    assert result["model"] == "patchcore"
    assert result["category"] == "wood"
    assert result["seed"] == 42
    assert result["n_test"] == 2
    assert result["config_path"] == "configs/run.yaml"

    assert result["image_auroc"] == pytest.approx(1.0)
    assert result["image_ap"] == pytest.approx(1.0)
    assert result["image_f1"] == pytest.approx(1.0)

    assert result["pixel_auroc"] == pytest.approx(1.0)
    assert result["pixel_dice"] == pytest.approx(1.0)
    assert result["pixel_iou"] == pytest.approx(1.0)

    assert result["threshold_image"] == pytest.approx(0.5)
    assert result["threshold_pixel"] == pytest.approx(0.5)
    assert result["threshold_source"] == "normal_validation"

    assert result["inference_ms_mean"] == pytest.approx(15.0)
    assert result["inference_ms_p95"] == pytest.approx(19.5)


def test_evaluate_predictions_rejects_empty_predictions(evaluator_case):
    evaluator_case["predictions"] = []

    with pytest.raises(ValueError):
        evaluate_predictions(
            **evaluator_case,
            threshold_pixel=0.5,
        )


def test_evaluate_predictions_rejects_duplicate_sample_ids(
    evaluator_case,
):
    duplicate = dict(evaluator_case["predictions"][0])
    evaluator_case["predictions"].append(duplicate)

    with pytest.raises(ValueError):
        evaluate_predictions(
            **evaluator_case,
            threshold_pixel=0.5,
        )


@pytest.mark.parametrize(
    ("field", "different_value"),
    [
        ("experiment_id", "run_002"),
        ("model", "padim"),
        ("category", "capsule"),
        ("seed", 123),
    ],
)
def test_evaluate_predictions_rejects_mixed_runs(
    evaluator_case,
    field,
    different_value,
):
    evaluator_case["predictions"][1][field] = different_value

    with pytest.raises(ValueError):
        evaluate_predictions(
            **evaluator_case,
            threshold_pixel=0.5,
        )


def test_evaluate_predictions_rejects_mixed_image_thresholds(
    evaluator_case,
):
    evaluator_case["predictions"][1]["threshold_image"] = 0.6

    with pytest.raises(ValueError):
        evaluate_predictions(
            **evaluator_case,
            threshold_pixel=0.5,
        )


def test_evaluate_predictions_requires_every_image_label(
    evaluator_case,
):
    evaluator_case["image_labels"].pop("defect_001")

    with pytest.raises(ValueError):
        evaluate_predictions(
            **evaluator_case,
            threshold_pixel=0.5,
        )


def test_evaluate_predictions_requires_every_ground_truth_mask(
    evaluator_case,
):
    evaluator_case["ground_truth_masks"].pop("defect_001")

    with pytest.raises(ValueError):
        evaluate_predictions(
            **evaluator_case,
            threshold_pixel=0.5,
        )


def test_evaluate_predictions_rejects_non_validation_source(
    evaluator_case,
):
    with pytest.raises(ValueError):
        evaluate_predictions(
            **evaluator_case,
            threshold_pixel=0.5,
            threshold_source="official_test",
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [float("nan"), float("inf"), float("-inf")],
)
def test_evaluate_predictions_rejects_invalid_pixel_threshold(
    evaluator_case,
    invalid_threshold,
):
    with pytest.raises(ValueError):
        evaluate_predictions(
            **evaluator_case,
            threshold_pixel=invalid_threshold,
        )


def test_evaluator_never_fits_threshold_from_test_data(
    evaluator_case,
    monkeypatch,
):
    import evaluation_research.thresholds as threshold_module

    def forbidden_threshold_fit(*args, **kwargs):
        raise AssertionError(
            "evaluator must not fit thresholds during final-test evaluation"
        )

    monkeypatch.setattr(
        threshold_module,
        "fit_normal_threshold",
        forbidden_threshold_fit,
    )
    monkeypatch.setattr(
        threshold_module,
        "fit_thresholds_from_normal_validation",
        forbidden_threshold_fit,
    )

    result = evaluate_predictions(
        **evaluator_case,
        threshold_pixel=0.5,
        threshold_source="normal_validation",
    )

    assert result["threshold_source"] == "normal_validation"
    assert result["threshold_pixel"] == pytest.approx(0.5)
