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

