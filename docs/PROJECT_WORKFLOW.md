# Project Workflow

## Giai đoạn 1 - Khóa protocol và interface

Người 1 chuẩn bị data loader, split, transform và config. Người 5 khóa metric/threshold protocol. Người 6 cùng hai owner này khóa shared inference/result schema. Sau thời điểm này, thay đổi interface phải qua Pull Request riêng.

Gate:

- official test không xuất hiện trong train/validation;
- seed và config được lưu;
- một dummy prediction đi qua evaluation và visualization thành công.

## Giai đoạn 2 - Phát triển model song song

Người 2, 3 và 4 làm việc trên branch riêng nhưng cùng dùng data API và output contract. Mỗi model phải chạy trên một category trước khi mở rộng.

Gate:

- có score và anomaly map;
- có config snapshot và artifact metadata;
- có sanity test;
- reviewer trong cặp đã duyệt.

## Giai đoạn 3 - Evaluation và research

Người 5 thu output đã freeze, chạy metric và ablation. Không cho mỗi model dùng cách threshold khác nhau nếu không được ghi rõ là phân tích phụ.

Gate:

- bảng metric theo category/model;
- tối thiểu 3 nhóm ablation;
- failure cases gồm FP, FN và localization error;
- model demo được chọn theo chất lượng + tốc độ + bộ nhớ.

## Giai đoạn 4 - Demo và báo cáo

Người 6 tích hợp model đã chọn qua adapter. Figure/bảng trong báo cáo phải truy ngược được về experiment id và config.

Gate:

- demo upload → score → heatmap → mask → overlay;
- README chạy lại được trên máy khác;
- experiment chính được rerun hoặc kiểm tra artifact đầy đủ;
- cả nhóm review kết quả cuối và rehearsal Q&A.

## Nhịp làm việc hằng ngày

1. Cập nhật Issue: `Todo` → `In Progress` → `In Review` → `Done`.
2. Pull `dev` trước khi bắt đầu thay đổi mới.
3. Push ít nhất một lần mỗi ngày làm việc để tránh code chỉ nằm trên máy cá nhân.
4. Không merge PR của chính mình khi chưa có approval.
5. Blocker quá một buổi làm việc phải ghi vào Issue và tag người phụ thuộc.

