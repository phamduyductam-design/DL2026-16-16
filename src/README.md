# Source ownership

- `datasets/`: Người 1 - loader, manifest, transforms, leakage guards.
- `models/`: Người 2-4 - Classical/Autoencoder, PaDiM, PatchCore.
- `evaluation/`: Người 5 - metric và threshold audit.
- `visualization/`: Người 6 - heatmap, binary mask và overlay.
- `utils/`: tiện ích dùng chung, thay đổi interface cần Leader review.

Mỗi model phải tuân thủ prediction interface trong README gốc để evaluation và demo không phụ thuộc implementation cụ thể.

