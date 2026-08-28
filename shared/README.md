# Shared Interface Contract

Owner phối hợp: **Người 1 + Người 5 + Người 6**.

## Model interface tối thiểu

Mỗi model adapter cần hỗ trợ logic tương đương:

```python
fit(train_loader, config) -> artifact
predict(image_or_batch, artifact, config) -> prediction
```

`prediction` phải có:

```json
{
  "experiment_id": "patchcore_wood_r256_seed42",
  "model": "patchcore",
  "category": "wood",
  "sample_id": "000123",
  "anomaly_score": 0.731,
  "threshold_image": 0.512,
  "predicted_label": "defect",
  "anomaly_map_path": "outputs/.../anomaly_map.npy",
  "binary_mask_path": "outputs/.../mask.png",
  "overlay_path": "outputs/.../overlay.png",
  "inference_ms": 18.4,
  "seed": 42,
  "config_path": "configs/patchcore_wood_r256.yaml"
}
```

## Quy tắc chung

- `anomaly_score` càng cao càng bất thường cho tất cả model.
- Anomaly map phải được đưa về cùng kích thước ảnh đánh giá trước khi tính pixel metric.
- Không ghi threshold được tối ưu trên test vào field `threshold_image` của kết quả chính.
- Mọi path trong CSV/JSON dùng đường dẫn tương đối từ repository hoặc artifact root.
- Thay đổi schema phải có PR riêng và approval của các owner phối hợp.

