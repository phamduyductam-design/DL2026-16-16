# GitHub Project Board - Task khởi tạo

Tạo board với các cột: `Backlog`, `Ready`, `In Progress`, `In Review`, `Blocked`, `Done`.

## Người 1 - Data & Protocol

- [ ] Chuẩn hóa cấu trúc MVTec AD và data path.
- [ ] Viết loader cho `wood`, `metal_nut`, `capsule`.
- [ ] Tạo deterministic normal validation split.
- [ ] Khóa preprocessing/augmentation config.
- [ ] Viết test chống data leakage.

## Người 2 - Baseline Models

- [ ] Classical CV baseline.
- [ ] Convolutional Autoencoder train/inference.
- [ ] Chuyển reconstruction error thành anomaly score/map.
- [ ] Xuất checkpoint và prediction schema chung.

## Người 3 - PaDiM

- [ ] Pretrained feature extractor.
- [ ] Gaussian statistics theo patch.
- [ ] Mahalanobis anomaly scoring.
- [ ] Thí nghiệm backbone và resolution.

## Người 4 - PatchCore

- [ ] Patch feature extractor.
- [ ] Memory bank và coreset sampler.
- [ ] Nearest-neighbor anomaly scoring.
- [ ] Thí nghiệm coreset ratio/backbone và đo resource.

## Người 5 - Evaluation & Research

- [ ] Image-level metrics.
- [ ] Pixel-level metrics và AUPRO/PRO.
- [ ] Validation-based threshold protocol.
- [ ] Ablation result aggregator.
- [ ] Error-analysis gallery.

## Người 6 - Deployment, Integration & Report

- [ ] Model adapter/registry.
- [ ] Web demo skeleton.
- [ ] Heatmap, mask và overlay rendering.
- [ ] End-to-end integration test.
- [ ] README tái lập, báo cáo và slide.

## Shared milestones

- [ ] W1 gate: data + baseline + protocol/metric freeze.
- [ ] W2 gate: PaDiM + PatchCore chạy trên 3 category.
- [ ] W3 gate: metric + ablation + error analysis + model selection.
- [ ] W4 gate: demo + report + rerun + rehearsal.

