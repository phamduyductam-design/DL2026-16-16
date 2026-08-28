# Người 2 - Baseline Model Lead

## Phạm vi

- Xây Classical CV baseline: Otsu/edge/morphology.
- Xây Convolutional Autoencoder train trên normal images.
- Sinh anomaly score cấp ảnh và anomaly map cấp pixel.
- Ghi nhận chất lượng, inference time và checkpoint Autoencoder.

## Deliverables

- Entry point train/inference cho Autoencoder.
- Classical baseline có config rõ ràng.
- Adapter xuất đúng shared output contract.
- Kết quả sơ bộ trên ít nhất 1 category ở tuần 1.
- Checkpoint và artifact chỉ lưu ngoài Git hoặc qua storage được nhóm thống nhất.

## Definition of Done

- Hai baseline chạy end-to-end từ config.
- Cùng một ảnh/config cho kết quả tái lập trong sai số cho phép.
- Người 5 cross-review metric và output schema.

Branch gợi ý: `feat/baseline/<task-name>`.

