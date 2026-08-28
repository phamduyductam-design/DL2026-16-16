# Người 5 - Evaluation & Research Lead

## Phạm vi

- Xây metric dùng chung: Image AUROC, AP, F1; Pixel AUROC, AUPRO/PRO, Dice, IoU.
- Khóa threshold protocol từ normal validation distribution.
- Tổng hợp so sánh model, ablation và practical metrics.
- Phân tích false positive, false negative và localization error.
- Kiểm tra tính công bằng giữa các model.

## Deliverables

- Evaluation entry point đọc shared result schema.
- CSV/JSON tổng hợp theo model, category, seed và experiment id.
- Bảng ablation tối thiểu 3 nhóm.
- Bộ failure cases kèm heatmap/overlay và nhận xét.
- Quy tắc tách main threshold khỏi oracle threshold nếu báo cả hai.

## Definition of Done

- Metric được test trên ví dụ nhỏ có kết quả biết trước.
- Không có bước tune bằng official test labels/masks.
- Người 2 cross-review metric và cách tổng hợp baseline.

Branch gợi ý: `feat/eval/<task-name>` hoặc `experiment/eval/<analysis>`.

