# Dữ liệu MVTec AD

Không upload dataset 5.27 GB lên GitHub.

Sau khi tải từ Kaggle, đặt hoặc liên kết dữ liệu theo cấu trúc:

```text
data/MVTecAD/
├── wood/
├── metal_nut/
├── capsule/
└── cable/
```

Mỗi category phải có `train/good`, `test/good`, `test/<defect_type>` và `ground_truth/<defect_type>`.

Đường dẫn thực tế được cấu hình trong `configs/data.yaml`. Tuyệt đối không đổi cấu trúc dataset hoặc sao chép ảnh test sang train.

