# Protocol dữ liệu và thực nghiệm

Tài liệu này do Người 1 - Data & Protocol Lead sở hữu. Mọi thay đổi phải được Người 1 duyệt trước khi merge.

## 1. Dataset

MVTec AD theo cấu trúc:

```text
MVTecAD/<category>/
├── train/good/*.png
├── test/good/*.png
├── test/<defect_type>/*.png
└── ground_truth/<defect_type>/*_mask.png
```

Category chính: `wood`, `metal_nut`, `capsule`; `cable` chỉ mở rộng sau khi ba category chính đã chạy end-to-end.

## 2. Quy tắc chia dữ liệu

| Split | Nguồn | Được dùng cho |
|---|---|---|
| Train | 80-85% official `train/good` | Học normality |
| Validation | 15-20% official `train/good`, seed cố định | Chọn hyperparameter và threshold |
| Test | Official `test/good` + `test/<defect>` | Đánh giá cuối |
| Ground truth | `ground_truth/<defect>` | Chỉ tính metric localization sau inference |

Mặc định dùng `seed: 42`. Danh sách file train/validation phải được lưu thành manifest để tất cả model dùng đúng một split.

## 3. Chống data leakage

- Không đưa ảnh lỗi thật hoặc mask test vào training.
- Không chọn backbone, resolution, augmentation, epoch, coreset ratio hay threshold dựa trên kết quả test cuối.
- Không dùng ground-truth test để chọn threshold chính.
- Nếu báo `oracle/best threshold`, phải tách khỏi kết quả chính và ghi rõ chỉ dùng phân tích.
- Không loại category hoặc run chỉ vì kết quả xấu.

## 4. Preprocessing

- Resolution cơ sở: `256 x 256`; ablation có thể so với `384 x 384`.
- Chuẩn hóa theo backbone pretrained.
- Augmentation nhẹ và phù hợp category: flip, rotation nhỏ, brightness/contrast nhẹ.
- Không mặc định dùng CutOut hoặc random erasing vì có thể tạo vùng giống lỗi.

## 5. Threshold

- Image threshold lấy từ percentile cố định trên anomaly score của validation normal.
- Pixel threshold lấy từ validation normal maps hoặc một quy tắc được khóa trước.
- Lưu threshold cùng category, model, seed và config hash.

## 6. Tái lập

Mỗi run phải lưu:

- `experiment_id`, timestamp, git commit SHA;
- seed, category, dataset manifest;
- transforms, resolution, backbone;
- batch size, learning rate, epoch nếu có;
- threshold rule;
- metric CSV/JSON;
- checkpoint, PaDiM statistics hoặc PatchCore memory bank tương ứng.

