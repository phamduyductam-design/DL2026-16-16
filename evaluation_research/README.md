# Người 5 — Evaluation & Research Lead

Module này cung cấp protocol và implementation dùng chung để đánh giá các model anomaly detection trong dự án.

## Phạm vi

- Image metrics: AUROC, AP và F1.
- Pixel metrics: AUROC, Dice, IoU và AUPRO@0.30.
- Threshold chỉ được fit từ normal validation.
- Kiểm tra shared prediction schema và artifact paths.
- Tổng hợp practical metrics.
- Xuất kết quả tự động sang CSV và JSON.

Primary metrics:

- Detection: `image_auroc`.
- Localization: `pixel_aupro_30`.

Các metric còn lại là supporting metrics.

## Protocol và config

Định nghĩa đầy đủ:

- [`METRIC_PROTOCOL.md`](METRIC_PROTOCOL.md)
- [`../configs/evaluation_protocol_v1.yaml`](../configs/evaluation_protocol_v1.yaml)
- [`../shared/README.md`](../shared/README.md)

Không được thay đổi threshold rule hoặc primary metric sau khi đã xem kết quả official final-test.

## Cấu trúc module

```text
evaluation_research/
├── metrics.py          # Image, pixel và AUPRO metrics
├── thresholds.py       # Threshold từ normal validation
├── schema.py           # Shared prediction validation
├── evaluator.py        # Entry point tổng hợp một experiment
├── result_writer.py    # CSV/JSON output
└── METRIC_PROTOCOL.md  # Đặc tả metric và chống leakage
```

## Luồng đánh giá

```text
normal validation scores
        |
        v
fit image/pixel thresholds
        |
        v
freeze threshold + config
        |
        v
model predictions on final-test
        |
        v
validate shared schema and artifacts
        |
        v
compute image/pixel/AUPRO metrics
        |
        v
write CSV and JSON summaries
```

Official test labels và masks chỉ xuất hiện ở bước tính metric cuối cùng. Chúng không được truyền vào hàm fit threshold.

## Fit threshold

```python
from evaluation_research import (
    fit_thresholds_from_normal_validation,
)

thresholds = fit_thresholds_from_normal_validation(
    normal_image_scores=normal_validation_image_scores,
    normal_pixel_scores=normal_validation_anomaly_maps,
    image_quantile=0.99,
    pixel_quantile=0.995,
    quantile_method="higher",
)

threshold_image = thresholds["threshold_image"]
threshold_pixel = thresholds["threshold_pixel"]
```

Hai threshold được fit độc lập. Model prediction phải ghi `threshold_image` đã đóng băng vào shared prediction record.

## Evaluate một experiment

```python
from evaluation_research import evaluate_predictions

summary = evaluate_predictions(
    predictions=prediction_records,
    image_labels=final_test_image_labels,
    ground_truth_masks=final_test_masks,
    artifact_root=".",
    threshold_pixel=threshold_pixel,
    threshold_source="normal_validation",
)
```

Yêu cầu:

- `predictions` chỉ chứa một `experiment_id`.
- Tất cả record phải cùng model, category, seed, config và image threshold.
- Mỗi `sample_id` phải duy nhất.
- Mỗi sample phải có image label và ground-truth mask.
- Anomaly map phải là NumPy array 2D, hữu hạn và cùng shape với mask.
- Các artifact path phải tương đối, tồn tại và không thoát khỏi `artifact_root`.

## Xuất kết quả

```python
from pathlib import Path

from evaluation_research import (
    write_results_csv,
    write_results_json,
)

output_dir = Path("outputs") / str(summary["experiment_id"])

write_results_csv(
    [summary],
    output_dir / "evaluation_summary.csv",
)
write_results_json(
    [summary],
    output_dir / "evaluation_summary.json",
)
```

Writer:

- Tự tạo parent directory.
- Không ghi đè file đã tồn tại.
- Chuyển metric NaN/infinity thành `null` trong JSON.
- Chuyển metric NaN/infinity thành ô trống trong CSV.
- Giữ thêm các field mở rộng như `warnings`.

Mỗi run phải dùng output path mới và `experiment_id` duy nhất.

## Trường kết quả chuẩn

```text
experiment_id
model
category
seed
n_test
image_auroc
image_ap
image_f1
pixel_auroc
pixel_aupro_30
pixel_dice
pixel_iou
threshold_image
threshold_pixel
threshold_source
inference_ms_mean
inference_ms_p95
config_path
```

## Chạy test

Kích hoạt virtual environment:

```bash
source .venv/bin/activate
```

Chạy test riêng cho evaluation:

```bash
python -m pytest -q tests/test_evaluation_*.py
```

Chạy toàn bộ repository:

```bash
python -m pytest -q
python -m compileall -q evaluation_research tests
```

Các test bao phủ:

- Toy examples có kết quả biết trước.
- Single-class và empty-mask edge cases.
- Threshold boundary dùng phép so sánh `>=`.
- Shape, binary-value và finite-value validation.
- Threshold chỉ dùng normal validation.
- Evaluation path không fit lại threshold bằng test labels/masks.
- Connected components 8-connectivity.
- AUPRO interpolation và background từ ảnh normal.
- CSV/JSON serialization và chống ghi đè.

## Quy trình review

Branch:

```text
feat/eval/metric-threshold-protocol
```

PR phải trỏ vào `dev`, liên kết `Closes #6`, chạy CI xanh và yêu cầu Người 2 review:

- Tính đúng của metric.
- Baseline compatibility.
- Threshold leakage.
- Comparison fairness.
- Result schema.