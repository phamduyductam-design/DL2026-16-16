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

