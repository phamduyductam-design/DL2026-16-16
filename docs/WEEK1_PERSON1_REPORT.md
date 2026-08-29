# Báo cáo hoàn thành tuần 1 — Người 1 / Leader

## Kết quả

Phần Data & Protocol tuần 1 đã hoàn thành trên ba category bắt buộc:
`wood`, `metal_nut`, `capsule`. `cable` chưa tải vì là phạm vi mở rộng.

Data root local:

```text
D:\deep data\data\mvtec_ad\
├── wood\
├── metal_nut\
├── capsule\
├── protocol\
│   ├── split_seed42.json
│   ├── split_seed42_evaluation_only.json
│   └── split_seed42_checksums_sha256.json
└── SOURCE_METADATA.md
```

## Số lượng đã xác minh

| Category | Official normal train | Protocol train | Normal validation | Official test | Test anomaly |
|---|---:|---:|---:|---:|---:|
| `wood` | 247 | 198 | 49 | 79 | 60 |
| `metal_nut` | 220 | 176 | 44 | 115 | 93 |
| `capsule` | 219 | 175 | 44 | 132 | 109 |
| **Tổng** | **686** | **549** | **137** | **326** | **262** |

Gói chọn lọc có 1.280 file thuộc ba category, tổng dung lượng
1.069.086.935 byte. Số này gồm ảnh train/test, mask, `readme.txt` và
`license.txt`; không gồm các protocol artifact sinh sau khi tải.

## Protocol đã khóa

- Seed: `42`.
- Validation fraction: `0.2`, tách riêng theo từng category.
- Xếp hạng sample bằng SHA-256 của `seed + sample_id`; split không phụ thuộc
  thứ tự filesystem.
- Resize: `256 × 256`.
- RGB normalization: ImageNet mean/std.
- Train augmentation: deterministic horizontal flip với xác suất `0.5`.
- Train và validation chỉ lấy từ official `train/good`.
- Official test không tham gia train, validation, config hoặc threshold tuning.
- Main split manifest không chứa defect label hoặc mask path.
- Test label/mask path chỉ nằm trong artifact có tên `evaluation_only`.

## Thành phần đã triển khai

- `MVTECADCatalog`: kiểm tra category, split, defect folder và cặp mask.
- `deterministic_normal_split`: tách normal validation tái lập được.
- `MVTECDataset`: trả image/category/sample id/source split; che test annotation
  ở inference mode.
- `MVTecTransform`: RGB → normalized `float32` CHW và binary mask resize bằng
  nearest-neighbor.
- `assert_protocol_integrity`: kiểm tra split disjoint, test leakage, duplicate
  sample id, mask contract và tùy chọn duplicate-content audit.
- `prepare_data.py`: CLI validate → split → audit → manifest.

## Kết quả kiểm thử

- 7/7 fixture tests pass.
- Toàn bộ ảnh và mask thật mở/giải mã được bằng Pillow.
- SHA-256 audit không phát hiện ảnh byte-identical nằm ở nhiều protocol split.
- SHA-256 inventory được ghi cho toàn bộ 1.280 file.
- Main manifest chạy lại với cùng config có nội dung byte-for-byte giống nhau.
- Mini-batch thật có shape `(2, 3, 256, 256)`, dtype `float32` và các field:
  `image`, `category`, `sample_id`, `source_split`, `image_path`.

## Handoff cho nhóm

Người 2–4 dùng:

- `configs/data_protocol_week1.yaml`;
- `data_protocol.MVTECADCatalog` để lấy record;
- `data_protocol.deterministic_normal_split` để tái tạo split;
- `data_protocol.MVTECDataset` và `data_protocol.MVTecTransform` cho input.

Không được đọc `split_seed42_evaluation_only.json` trong train, fit, inference,
model selection hoặc threshold tuning. File đó dành cho evaluation path của
Người 5.

## Rủi ro cần leader theo dõi

- Dataset được nhận qua Kaggle mirror, không phải tải trực tiếp từ form chính
  thức của MVTec. Structure/count/decoding/checksum đã được audit nhưng không có
  official checksum để chứng minh mirror giống tuyệt đối bản gốc.
- Kaggle Data Card hiển thị license là `Unknown`, trong khi file nhúng và trang
  MVTec chính thức ghi CC BY-NC-SA 4.0. Nhóm phải giữ attribution và chỉ dùng
  theo phạm vi phi thương mại của giấy phép.
- PyTorch/CUDA chưa được pin trong issue này. Nhóm cần khóa runtime chung trước
  khi Người 2–4 thêm `torch`/`torchvision` để tránh cài nhầm CUDA build.

## Gate thuộc Người 1

- [x] Ba category bắt buộc load được.
- [x] Deterministic normal validation split được sinh và lưu.
- [x] Preprocessing, augmentation, seed và config được khóa.
- [x] Official test không xuất hiện trong train/validation.
- [x] Ground-truth mask được tách khỏi inference path.
- [x] Structure, decode, leakage và checksum audits chạy thành công.
- [x] README, source/license note, CLI và test đã hoàn thiện.
- [ ] Người 4 cross-review và approval PR — cần thành viên thực hiện, leader
  không tự approval thay reviewer.
