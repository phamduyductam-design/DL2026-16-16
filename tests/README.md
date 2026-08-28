# Tests

Ưu tiên các test có giá trị bảo vệ protocol và integration:

- loader không đưa official test vào train/validation;
- validation split tái lập theo seed;
- mask chỉ được đọc trong evaluation path;
- mọi model trả đúng shared output schema;
- anomaly map và ground-truth có cùng kích thước khi tính metric;
- metric cho kết quả đúng trên toy arrays;
- demo adapter xử lý normal/defect prediction và input lỗi.

