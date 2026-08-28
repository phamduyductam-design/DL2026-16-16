# Industrial Surface Anomaly Detection

Đồ án Deep Learning phát hiện bất thường ở cấp ảnh và định vị vùng lỗi ở cấp pixel trên MVTec AD.

## Phạm vi đã khóa

- Category chính: `wood`, `metal_nut`, `capsule`.
- Category mở rộng: `cable` nếu đủ thời gian.
- Mô hình: Classical CV, Convolutional Autoencoder, PaDiM và PatchCore.
- PatchCore là mô hình chính; PaDiM là đối chứng mạnh; Autoencoder là Deep Learning baseline.
- Train chỉ dùng ảnh `train/good`; test image và ground-truth mask không được dùng để huấn luyện hay chọn mô hình.

## Bắt đầu nhanh

1. Đọc [`docs/PROTOCOL.md`](docs/PROTOCOL.md) trước khi viết code.
2. Đặt MVTec AD ngoài Git và cấu hình đường dẫn trong `configs/data.yaml`.
3. Mỗi thành viên tạo branch theo [`CONTRIBUTING.md`](CONTRIBUTING.md).
4. Làm việc theo [`docs/WORKFLOW.md`](docs/WORKFLOW.md) và backlog trong [`docs/TASKS.md`](docs/TASKS.md).
5. Chỉ merge khi PR đạt Definition of Done và đã qua cross-review.

## Cấu trúc repository

```text
deep_learning/
├── .github/                 # CI, issue và pull-request template
├── configs/                 # Cấu hình dữ liệu/thí nghiệm
├── data/                    # Chỉ hướng dẫn, không commit dataset
├── docs/                    # Workflow, protocol và backlog
├── src/
│   ├── datasets/            # Người 1
│   ├── models/              # Người 2, 3, 4
│   ├── evaluation/          # Người 5
│   ├── visualization/       # Người 6
│   └── utils/               # Tiện ích dùng chung
├── scripts/                 # Entry points train/fit/evaluate
├── app/                     # Web demo
├── outputs/                 # Kết quả tái tạo được; file lớn không commit
└── tests/                   # Sanity/unit tests
```

## Interface bàn giao bắt buộc

Dataset loader phải trả tối thiểu:

```python
{
    "image": image_tensor,
    "label": 0_or_1,
    "mask": mask_tensor,
    "category": category,
    "defect_type": defect_type,
    "image_path": image_path,
}
```

Mỗi model khi inference phải trả tối thiểu:

```python
{
    "image_score": float_score,
    "anomaly_map": anomaly_map,
    "metadata": {"model": model_name, "category": category},
}
```

## Definition of Done toàn dự án

- Không có data leakage.
- Cùng split, transform và metric cho các model so sánh.
- Có AUROC/AP ở cấp ảnh; Pixel AUROC/AUPRO và Dice/IoU ở cấp pixel.
- Có ít nhất ba nhóm ablation và error analysis.
- Lưu seed, config, version thư viện và kết quả CSV/JSON.
- Demo hiển thị anomaly score, Good/Defect, heatmap, mask và overlay.

