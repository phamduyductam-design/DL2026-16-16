import csv
import json

import numpy as np
import pytest

from evaluation_research import (
    evaluate_predictions,
    fit_thresholds_from_normal_validation,
    write_results_csv,
    write_results_json,
)


def _create_prediction(
    artifact_root,
    *,
    sample_id,
    anomaly_score,
    anomaly_map,
    threshold_image,
    inference_ms,
):
    output_dir = artifact_root / "outputs" / sample_id
    output_dir.mkdir(parents=True)

    anomaly_map_path = output_dir / "anomaly_map.npy"
    binary_mask_path = output_dir / "mask.png"
    overlay_path = output_dir / "overlay.png"

    np.save(anomaly_map_path, np.asarray(anomaly_map, dtype=float))
    binary_mask_path.write_bytes(b"mask fixture")
    overlay_path.write_bytes(b"overlay fixture")

    return {
        "experiment_id": "evaluation_e2e_seed42",
        "model": "dummy",
        "category": "wood",
        "sample_id": sample_id,
        "anomaly_score": anomaly_score,
        "threshold_image": threshold_image,
        "predicted_label": (
            "defect"
            if anomaly_score >= threshold_image
            else "normal"
        ),
        "anomaly_map_path": str(
            anomaly_map_path.relative_to(artifact_root)
        ),
        "binary_mask_path": str(
            binary_mask_path.relative_to(artifact_root)
        ),
        "overlay_path": str(
            overlay_path.relative_to(artifact_root)
        ),
        "inference_ms": inference_ms,
        "seed": 42,
        "config_path": "configs/evaluation_protocol_v1.yaml",
    }


def test_complete_evaluation_pipeline(tmp_path):
    artifact_root = tmp_path / "repository"
    config_dir = artifact_root / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "evaluation_protocol_v1.yaml").write_text(
        "protocol_version: 1\n",
        encoding="utf-8",
    )

    thresholds = fit_thresholds_from_normal_validation(
        normal_image_scores=[0.1, 0.2],
        normal_pixel_scores=[
            [[0.01, 0.02], [0.03, 0.1]],
        ],
        image_quantile=0.99,
        pixel_quantile=0.995,
        quantile_method="higher",
    )

    threshold_image = float(thresholds["threshold_image"])
    threshold_pixel = float(thresholds["threshold_pixel"])

    predictions = [
        _create_prediction(
            artifact_root,
            sample_id="normal_001",
            anomaly_score=0.1,
            anomaly_map=[[0.01, 0.01], [0.01, 0.01]],
            threshold_image=threshold_image,
            inference_ms=10.0,
        ),
        _create_prediction(
            artifact_root,
            sample_id="defect_001",
            anomaly_score=0.9,
            anomaly_map=[[0.9, 0.01], [0.01, 0.01]],
            threshold_image=threshold_image,
            inference_ms=20.0,
        ),
    ]

    summary = evaluate_predictions(
        predictions=predictions,
        image_labels={
            "normal_001": 0,
            "defect_001": 1,
        },
        ground_truth_masks={
            "normal_001": np.zeros((2, 2), dtype=np.uint8),
            "defect_001": np.array(
                [[1, 0], [0, 0]],
                dtype=np.uint8,
            ),
        },
        artifact_root=artifact_root,
        threshold_pixel=threshold_pixel,
        threshold_source=str(thresholds["threshold_source"]),
    )

    assert summary["image_auroc"] == pytest.approx(1.0)
    assert summary["image_ap"] == pytest.approx(1.0)
    assert summary["image_f1"] == pytest.approx(1.0)
    assert summary["pixel_auroc"] == pytest.approx(1.0)
    assert summary["pixel_aupro_30"] == pytest.approx(1.0)
    assert summary["pixel_dice"] == pytest.approx(1.0)
    assert summary["pixel_iou"] == pytest.approx(1.0)

    result_dir = artifact_root / "outputs" / "evaluation_e2e_seed42"
    csv_path = write_results_csv(
        [summary],
        result_dir / "summary.csv",
    )
    json_path = write_results_json(
        [summary],
        result_dir / "summary.json",
    )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))

    json_rows = json.loads(json_path.read_text(encoding="utf-8"))

    assert len(csv_rows) == 1
    assert len(json_rows) == 1
    assert csv_rows[0]["experiment_id"] == "evaluation_e2e_seed42"
    assert json_rows[0]["experiment_id"] == "evaluation_e2e_seed42"
    assert json_rows[0]["threshold_source"] == "normal_validation"
    assert json_rows[0]["pixel_aupro_30"] == pytest.approx(1.0)
