# Người 6 - Deployment, Integration & Report Lead

## Phạm vi

- Chuẩn hóa inference adapter cho model được chọn.
- Tích hợp web demo bằng Streamlit hoặc Gradio.
- Hiển thị upload image, anomaly score, Good/Defect, heatmap, mask và overlay.
- Kiểm thử end-to-end.
- Quản lý README, hình/bảng kết quả, báo cáo và slide.

## Deliverables

- Demo entry point và hướng dẫn chạy local.
- Model registry/adapter không phụ thuộc chi tiết nội bộ của từng model.
- Figure được sinh từ output có truy vết experiment id.
- README tái lập được cho train/fit/evaluate/demo.
- Checklist rehearsal Q&A và bản báo cáo/slide cuối.

## Definition of Done

- Demo xử lý được ảnh hợp lệ và báo lỗi thân thiện cho input không hợp lệ.
- Model được freeze từ đầu tuần 4; demo không tự tune threshold.
- Người 3 cross-review model adapter và visualization.

Branch gợi ý: `feat/demo/<task-name>` hoặc `docs/report/<section>`.

