# Người 4 - PatchCore Model Lead

## Phạm vi

- Trích xuất patch-level pretrained features.
- Xây memory bank và coreset subsampling.
- Tính nearest-neighbor anomaly score/map.
- So sánh backbone, resolution và coreset ratio.
- Theo dõi RAM, artifact size và inference time.

## Deliverables

- Fit/inference entry point cho PatchCore.
- Memory bank/coreset artifact có metadata đầy đủ.
- Output theo shared contract.
- Bảng trade-off chất lượng, tốc độ và bộ nhớ.

## Definition of Done

- Chạy được trên ít nhất 1 category ở tuần 2, sau đó mở rộng 3 category.
- Coreset ratio và backbone được điều khiển bằng config.
- Người 1 cross-review data assumptions và chống leakage.

Branch gợi ý: `feat/patchcore/<task-name>` hoặc `experiment/patchcore/<variant>`.

