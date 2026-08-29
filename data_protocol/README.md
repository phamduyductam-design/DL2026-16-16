# Người 1 - Data & Protocol Lead

## Phạm vi

- Tổ chức MVTec AD và kiểm tra cấu trúc category.
- Viết dataset loader và deterministic normal validation split.
- Chuẩn hóa resize, normalization và augmentation nhẹ.
- Quản lý config, seed và quy tắc chống data leakage.
- Bàn giao data API thống nhất cho Người 2-4.

## Deliverables

- Loader trả image, category, sample id và metadata cần thiết.
- Split manifest có seed, không trộn official test vào train/validation.
- Transform config tái lập được.
- Sanity test cho số lượng mẫu, mask và split.
- Tài liệu cách chuẩn bị dữ liệu local.

## Definition of Done

- Ba category chính `wood`, `metal_nut`, `capsule` load được.
- Validation chỉ lấy từ normal training images.
- Ground-truth mask chỉ được load trong evaluation path.
- Người 4 cross-review và xác nhận không có leakage.

Branch gợi ý: `feat/data/<task-name>`.

## Implementation tuần 1

API chính nằm trong package `data_protocol`:

```python
from data_protocol import (
    MVTECADCatalog,
    MVTECDataset,
    MVTecTransform,
    TransformConfig,
    deterministic_normal_split,
)

official_train, official_test = MVTECADCatalog(
    data_root="D:/deep data/data/mvtec_ad",
    categories=("wood", "metal_nut", "capsule"),
).scan()

train, validation = deterministic_normal_split(
    official_train,
    validation_fraction=0.2,
    seed=42,
)

transform = MVTecTransform(TransformConfig(height=256, width=256, seed=42))
train_dataset = MVTECDataset(train, transform=transform, training=True)
test_inference_dataset = MVTECDataset(official_test, transform=transform)
test_evaluation_dataset = MVTECDataset(
    official_test,
    transform=transform,
    evaluation=True,
)
```

`test_inference_dataset` không trả `is_anomaly`, `defect_type` hoặc mask.
Annotation chính thức chỉ xuất hiện trong `test_evaluation_dataset`.

### Tạo manifest và chạy audit

```powershell
python scripts/prepare_data.py `
  --config configs/data_protocol_week1.yaml `
  --data-root 'D:\deep data\data\mvtec_ad' `
  --output outputs/data_protocol/split_seed42.json `
  --verify-images `
  --hash-audit
```

`--verify-images` dùng Pillow để kiểm tra toàn bộ file ảnh/mask. `--hash-audit`
tính SHA-256, dừng nếu có ảnh byte-identical xuất hiện ở nhiều protocol split
và ghi sidecar `split_seed42_checksums_sha256.json` cho toàn bộ file thuộc ba
category đã chọn.

CLI luôn tách annotation chính thức sang
`split_seed42_evaluation_only.json`. Manifest chính chỉ chứa test image path và
không có `is_anomaly`, `defect_type` hoặc `mask_path`; model owner không được
đọc file `evaluation_only` trong fit, inference hoặc threshold tuning.

### Chạy test

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
```

Config tuần 1 dùng JSON syntax hợp lệ theo YAML 1.2, vì vậy code đọc được bằng
standard library và các model owner vẫn có thể dùng YAML tooling thông thường.
