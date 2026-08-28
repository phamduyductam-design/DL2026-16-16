## Mục tiêu

Mô tả ngắn thay đổi và lý do cần thay đổi.

Closes #<issue-number>

## Workstream

- [ ] Data & Protocol
- [ ] Baseline Models
- [ ] PaDiM
- [ ] PatchCore
- [ ] Evaluation & Research
- [ ] Deployment / Integration / Report
- [ ] Shared / CI / Documentation

## Cách kiểm tra

Ghi lệnh chạy, config, category, seed và output mẫu.

## Checklist

- [ ] Branch được tạo từ `dev` và PR trỏ vào `dev`.
- [ ] Không dùng test label/mask để train, tune hoặc chọn threshold.
- [ ] Output tuân theo interface chung.
- [ ] Có test hoặc sanity check phù hợp.
- [ ] Không commit dataset, checkpoint, memory bank hoặc secret.
- [ ] README/config đã cập nhật nếu cách chạy hoặc interface thay đổi.
- [ ] Đã tự review diff trước khi yêu cầu reviewer.

## Kết quả / ảnh minh họa

Đính kèm metric, log ngắn, heatmap/overlay hoặc ảnh demo khi phù hợp.

## Reviewer

Gắn reviewer theo cặp cross-review: 1↔4, 2↔5, 3↔6.

