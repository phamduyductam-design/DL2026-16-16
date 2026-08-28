# Experiment Outputs

Không commit artifact lớn vào GitHub. Chỉ commit các bảng/figure nhỏ cần cho báo cáo khi nhóm đã review.

Cấu trúc đề xuất:

```text
outputs/
└── <model>/
    └── <category>/
        └── <experiment_id>/
            ├── config.snapshot.yaml
            ├── metrics.json
            ├── predictions.csv
            ├── figures/
            └── run_metadata.json
```

Checkpoint, PaDiM statistics và PatchCore memory bank nên lưu ở artifact storage dùng chung; README/report chỉ trỏ tới artifact id hoặc release phù hợp.

