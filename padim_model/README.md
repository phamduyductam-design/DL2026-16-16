# Người 3 - PaDiM Model Lead

## Phạm vi

- Trích xuất pretrained CNN features.
- Xây Gaussian statistics theo patch/vị trí.
- Tính Mahalanobis distance để sinh anomaly map và image score.
- Chạy so sánh backbone và input resolution theo config chung.

## Deliverables

- Fit/inference entry point cho PaDiM.
- Feature/statistics artifact có metadata category và config.
- Output theo shared contract.
- Báo cáo thời gian fit/inference và bộ nhớ nếu đo được.

## Definition of Done

- Chạy được trên ít nhất 1 category ở tuần 2, sau đó mở rộng 3 category.
- Không dùng test label/mask trong quá trình fit hoặc chọn tham số.
- Người 6 cross-review inference adapter và visualization output.

Branch gợi ý: `feat/padim/<task-name>` hoặc `experiment/padim/<variant>`.

