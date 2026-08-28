# Người 1 - Data & Protocol Lead

Khu vực của Leader, chịu trách nhiệm tổ chức MVTec AD, khóa split, preprocessing,
contracts và kiểm soát data leakage cho toàn nhóm.

Đầu ra bàn giao:

- `Sample` và `Prediction` contracts dùng chung;
- split train/validation tái lập bằng seed 42;
- kiểm tra train/validation/test không chồng chéo;
- manifest file cho từng category;
- transforms và cấu hình dữ liệu thống nhất.

