# Quy trình đóng góp

## 1. Branch model

- `main`: phiên bản ổn định, chỉ nhận code đã tích hợp và review.
- `dev`: nhánh tích hợp hằng ngày của nhóm.
- Nhánh công việc: tạo từ `dev`, không tạo trực tiếp từ `main`.

Quy tắc đặt tên:

```text
<type>/<scope>/<short-description>
```

`type`: `feat`, `fix`, `experiment`, `docs`, `test`, `chore`.

`scope`: `data`, `baseline`, `padim`, `patchcore`, `eval`, `demo`, `report`, `shared`, `ci`.

Ví dụ:

```text
feat/data/mvtec-loader
experiment/padim/resnet18-384
fix/eval/normal-threshold
docs/report/error-analysis
```

## 2. Commit convention

```text
<type>(<scope>): <mô tả ngắn ở thể mệnh lệnh>
```

Ví dụ:

```text
feat(data): add deterministic normal validation split
feat(baseline): export autoencoder anomaly maps
experiment(patchcore): compare coreset ratios
fix(eval): prevent test-mask access during threshold selection
docs(report): add failure-case discussion
```

Mỗi commit chỉ nên chứa một thay đổi logic. Không commit dataset, checkpoint lớn, memory bank, cache hoặc secret.

## 3. Quy trình làm một task

1. Tạo hoặc nhận một GitHub Issue có mục tiêu và Definition of Done rõ ràng.
2. Đồng bộ `dev` và tạo branch công việc.
3. Viết code trong đúng workstream; phần dùng chung đặt tại `shared/`.
4. Chạy sanity test và tạo output mẫu nhỏ.
5. Cập nhật README/config nếu interface hoặc cách chạy thay đổi.
6. Mở Pull Request vào `dev` bằng template có sẵn.
7. Sửa tất cả review blocker và chờ CI xanh trước khi merge.

## 4. Review ownership

| Tác giả chính | Reviewer tối thiểu | Nội dung reviewer phải kiểm tra |
|---|---|---|
| Người 1 | Người 4 | Split, leakage, transform, config, seed |
| Người 2 | Người 5 | Output contract, metric baseline, reproducibility |
| Người 3 | Người 6 | Inference interface, artifact và visualization |
| Người 4 | Người 1 | Data assumptions, memory bank, config, resource usage |
| Người 5 | Người 2 | Metric, threshold, comparison fairness, result schema |
| Người 6 | Người 3 | Model adapter, demo flow, report figures |

PR ảnh hưởng đến interface chung phải có thêm review của Người 1, Người 5 hoặc Người 6 tùy phạm vi.

## 5. Definition of Done cho mọi PR

- [ ] Liên kết đúng Issue.
- [ ] Không truy cập test label/mask trong train, validation hoặc threshold tuning.
- [ ] Có config và seed cho experiment liên quan.
- [ ] Output tuân theo `shared/README.md`.
- [ ] Không commit dữ liệu hoặc artifact lớn.
- [ ] Có test/sanity check phù hợp.
- [ ] README hoặc hướng dẫn chạy đã được cập nhật.
- [ ] CI xanh và có ít nhất 1 approval.

## 6. Quy tắc kết quả thực nghiệm

- Không sửa CSV/JSON kết quả bằng tay nếu có thể sinh tự động.
- Mỗi run dùng một `experiment_id` duy nhất.
- Không ghi đè run cũ; tạo run mới khi config thay đổi.
- Kết quả chính phải dùng threshold từ normal validation distribution.
- Nếu báo oracle threshold, đặt field/column riêng và không dùng làm kết luận chính.
- Với run quan trọng có stochastic training, ưu tiên ghi mean ± standard deviation khi đủ thời gian.

