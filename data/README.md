# Dataset

Không commit ảnh MVTec AD vào GitHub.

Cấu trúc local đã khóa cho tuần 1:

```text
data/
└── mvtec_ad/
    ├── wood/
    ├── metal_nut/
    ├── capsule/
    └── cable/        # tùy chọn, chỉ tải khi nhóm quyết định mở rộng
```

Ba category bắt buộc là `wood`, `metal_nut`, `capsule`. Không cần tải toàn bộ
15 category của MVTec AD để hoàn thành mốc hiện tại.

## Data root

Code không hard-code đường dẫn máy cá nhân. Truyền `--data-root` hoặc đặt biến
môi trường `MVTEC_AD_ROOT` trỏ tới thư mục chứa các category. Ví dụ PowerShell:

```powershell
$env:MVTEC_AD_ROOT = 'D:\deep data\data\mvtec_ad'
python scripts/prepare_data.py --verify-images --hash-audit
```

Nếu data root còn một lớp `MVTecAD/`, loader sẽ tự nhận diện để tương thích với
cấu trúc tải trực tiếp từ KaggleHub.

## Quy tắc bắt buộc

- `train/good` là nguồn duy nhất cho protocol train và validation.
- Official `test/` không được đưa vào train/validation.
- `ground_truth/` chỉ được đọc khi dataset chạy ở `evaluation=True`.
- Split manifest chính chỉ lưu đường dẫn tương đối, seed và config; test label,
  defect type và mask path nằm trong manifest `evaluation_only` riêng.
- Giữ nguyên `readme.txt` và `license.txt` đi kèm mỗi category.

Nguồn, giấy phép, citation và cảnh báo provenance được ghi tại
[`docs/DATASET_SOURCE.md`](../docs/DATASET_SOURCE.md).
