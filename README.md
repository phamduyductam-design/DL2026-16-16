# Industrial Surface Anomaly Detection

Hệ thống phát hiện bất thường cấp ảnh và định vị vùng lỗi cấp pixel trên MVTec AD.
Repository được chia thành sáu khu vực chức năng độc lập để sáu thành viên làm song song.

## Cấu trúc chính

| Khu vực | Phụ trách | Nội dung |
|---|---|---|
| [`data_protocol/`](data_protocol/) | Người 1 - Leader | Dataset, split, transforms, contracts, chống leakage |
| [`baselines/`](baselines/) | Người 2 | Classical CV và Convolutional Autoencoder |
| [`padim/`](padim/) | Người 3 | Gaussian feature statistics và Mahalanobis scoring |
| [`patchcore/`](patchcore/) | Người 4 | Memory bank, coreset và nearest-neighbor scoring |
| [`evaluation/`](evaluation/) | Người 5 | Metric, threshold, ablation và error analysis |
| [`deployment/`](deployment/) | Người 6 | Inference adapter, visualization và web demo |

Các phần dùng chung nằm ở `.github/`, `configs/`, `docs/`, `scripts/` và `tests/`.

## Cây thư mục

```text
deep_learning/
├── .github/                 # CI và template làm việc
├── baselines/               # Người 2
├── configs/                 # Cấu hình dùng chung
├── data_protocol/           # Người 1
├── deployment/              # Người 6
├── docs/                    # Protocol, workflow và phân công
├── evaluation/              # Người 5
├── padim/                   # Người 3
├── patchcore/               # Người 4
├── scripts/                 # Lệnh train/fit/evaluate/demo
├── tests/                   # Kiểm tra contract và protocol
├── .env.example
├── .gitignore
├── .python-version
├── CONTRIBUTING.md
├── Makefile
├── pyproject.toml
├── requirements-dev.txt
└── requirements.txt
```

## Chạy nhanh

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
make check
```

Dataset không được commit. Khai báo đường dẫn bằng biến `MVTEC_ROOT` hoặc trong
`configs/base.yaml`.

## Protocol đã khóa

- Category chính: `wood`, `metal_nut`, `capsule`; `cable` là mở rộng.
- Train chỉ dùng official `train/good`.
- Validation tách 20% từ normal train với seed 42.
- Official test và ground-truth masks chỉ dùng đánh giá cuối.
- Threshold lấy từ validation normal, không tối ưu trên test.
- Tất cả model dùng chung `Sample` và `Prediction` contracts.

Xem [`docs/TEAM.md`](docs/TEAM.md), [`docs/PROTOCOL.md`](docs/PROTOCOL.md) và
[`docs/WORKFLOW.md`](docs/WORKFLOW.md) trước khi bắt đầu.
