# Entry-point Scripts

Thư mục này chứa CLI mỏng, gọi implementation từ từng workstream.

Tên gợi ý:

```text
prepare_data.py
train_autoencoder.py
fit_anomaly_model.py
evaluate.py
run_demo.py
```

CLI phải nhận config/path bằng argument hoặc environment variable; không hard-code đường dẫn máy cá nhân. Logic model chính không nên nằm trong script entry point.

## Entry point hiện có

`prepare_data.py` thực hiện bốn bước:

1. xác thực cấu trúc category và cặp test image/ground-truth mask;
2. tách deterministic normal validation từ official training set;
3. kiểm tra leakage theo path và tùy chọn theo SHA-256;
4. ghi split manifest có config, seed, source metadata và đường dẫn tương đối.

Chạy `python scripts/prepare_data.py --help` để xem đầy đủ tham số.
