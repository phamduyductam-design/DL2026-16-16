# Evaluation Metric and Threshold Protocol

Status: Draft v0.1 — khóa trước khi xem kết quả final-test.

## 1. Phạm vi và quy ước

- Nhãn `0` là normal, nhãn `1` là defect.
- `anomaly_score` và giá trị anomaly map càng cao thì càng bất thường.
- Mọi score và anomaly map phải là số hữu hạn, không chứa NaN hoặc infinity.
- Tất cả metric trong tài liệu này được tính trên official final-test set sau khi threshold đã được đóng băng từ normal validation.
- Không dùng final-test label hoặc mask để chọn threshold, hyperparameter hay model.
- Theo shared contract hiện tại, `experiment_id` là định danh duy nhất cho một tổ hợp model, category, config và seed.

## 2. Primary metrics

Các metric phải được đăng ký trước khi xem kết quả:

- Primary detection metric: Image AUROC.
- Primary localization metric: Pixel AUPRO@0.30.
- AP, F1, Pixel AUROC, Dice và IoU là supporting metrics.

Không được đổi primary metric sau khi xem kết quả để ưu tiên một model cụ thể.

## 3. Image-level metrics

### Image AUROC

Tính AUROC từ nhãn ảnh và anomaly score liên tục. Metric này không dùng threshold.

### Average Precision

Tính AP từ nhãn ảnh và anomaly score liên tục. Metric này không dùng threshold.

### Image F1

Tạo predicted label bằng quy tắc:

`predicted_label = anomaly_score >= threshold_image`

Sau đó tính F1 với defect là positive class.

Nếu tập đánh giá không chứa đủ cả normal và defect, cả ba metric `image_auroc`, `image_ap` và `image_f1` đều được xem là không xác định. Evaluator phải trả NaN cho cả ba metric và phát `RuntimeWarning` chứa cụm từ `single class`, thay vì crash hoặc âm thầm trả kết quả gây hiểu nhầm.
## 4. Pixel-level metrics

Anomaly map và ground-truth mask phải có cùng kích thước trước khi tính metric.

- Không được âm thầm resize ground-truth mask.
- Nếu cần resize, chỉ resize prediction map về kích thước đánh giá.
- Score map dùng nội suy bilinear.
- Binary mask dùng nearest-neighbor.

### Pixel AUROC

Flatten anomaly maps và ground-truth masks trong từng category, sau đó tính AUROC trên toàn bộ pixels.

### Dice và IoU

Protocol sử dụng micro-average ở cấp category.

TP, FP và FN được cộng gộp trên toàn bộ ảnh của một category trước khi tính:

`Dice = 2 * TP / (2 * TP + FP + FN)`

`IoU = TP / (TP + FP + FN)`

Quy tắc edge case áp dụng cho toàn bộ category:

- Nếu tổng ground truth và tổng prediction đều không có positive pixel, Dice và IoU bằng 1.
- Nếu chỉ một phía có positive pixel, Dice và IoU bằng 0.

### AUPRO@0.30

- Tách defect regions trong ground-truth mask bằng connected components với 8-connectivity.
- Với mỗi threshold, tính overlap riêng cho từng defect region.
- PRO là trung bình overlap của tất cả defect regions có thật.
- FPR được tính trên tất cả pixels không thuộc defect region trong toàn bộ final-test set, bao gồm background của ảnh defect và toàn bộ pixels của ảnh normal.
- Dùng một threshold grid xác định trước và ghi lại trong config.
- Chỉ giữ phần đường cong có `FPR <= 0.30`.
- Nội suy điểm biên tại `FPR = 0.30` nếu cần.
- Tích phân đường cong PRO–FPR và chia cho `0.30`.
- Tên trường kết quả là `pixel_aupro_30`.
- Nếu tập đánh giá không có defect region, trả NaN và cảnh báo rõ.

Cấu hình mặc định:

```yaml
aupro:
  max_fpr: 0.30
  num_thresholds: 200
  connectivity: 8
```

## 5. Threshold protocol

Image threshold và pixel threshold được ước lượng độc lập vì chúng có phân phối khác nhau.

```yaml
threshold_image:
  source: normal_validation
  method: quantile
  quantile: 0.99
  quantile_method: higher

threshold_pixel:
  source: normal_validation
  method: quantile
  quantile: 0.995
  quantile_method: higher
```

Quy tắc bắt buộc:

- `threshold_image` chỉ được fit từ image scores của normal validation.
- `threshold_pixel` chỉ được fit từ anomaly-map scores của normal validation.
- Hai giá trị quantile phải được khóa trước khi xem final-test result.
- Không thử nhiều quantile rồi chọn giá trị tốt nhất trên final-test.
- Test labels và test masks không được truyền vào hàm fit threshold.
- Cùng normal validation input và config phải sinh cùng threshold.
- Nếu pixel scores được sampling để tiết kiệm bộ nhớ, sampling phải deterministic và lưu seed cùng số lượng pixels đã dùng.

## 6. Main và oracle threshold

Kết quả chính luôn sử dụng normal-validation threshold.

Nếu nhóm báo oracle threshold, phải đặt tên riêng:

- `threshold_image_main`
- `threshold_image_oracle`
- `image_f1_main`
- `image_f1_oracle`
- `threshold_pixel_main`
- `threshold_pixel_oracle`
- `pixel_dice_main`
- `pixel_dice_oracle`

Oracle chỉ là phân tích phụ. Không được dùng oracle để chọn model, hyperparameter hoặc viết kết luận chính.

## 7. Aggregation

- Metric được tính riêng theo `experiment_id`, model, category và seed.
- Không pool dữ liệu từ nhiều category trước khi tính metric.
- Kết quả tổng hợp nhiều category dùng macro-average theo category.
- Nhiều seed được báo cáo bằng mean và sample standard deviation.
- Nếu chỉ có một seed, standard deviation được ghi là NaN thay vì 0.
- Không làm tròn trong quá trình tính toán; chỉ làm tròn khi xuất bảng báo cáo.

## 8. Warnings và lỗi

Evaluator phải:

- Phát cảnh báo cho metric không xác định do single-class hoặc không có defect region.
- Báo lỗi nếu score hoặc anomaly map chứa NaN/infinity.
- Báo lỗi nếu prediction và ground-truth mask sai shape.
- Báo lỗi nếu thiếu field bắt buộc trong shared prediction schema.
- Báo lỗi nếu artifact path là đường dẫn tuyệt đối, thoát khỏi artifact root hoặc không tồn tại.
