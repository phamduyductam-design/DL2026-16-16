# Bộ issue tuần 1 cho 6 thành viên

Các issue dưới đây là bản nháp để duyệt trước khi tạo trên GitHub.

Quy ước chung:

- Milestone: `Week 1 - Data, Baseline & Protocol Freeze`.
- Trạng thái ban đầu: `Ready`.
- Mỗi branch được tạo từ `dev`; Pull Request mở vào `dev`.
- Mọi Pull Request phải đáp ứng Definition of Done chung trong `CONTRIBUTING.md`.
- Không dùng official test labels hoặc ground-truth masks để train, chọn hyperparameter hay threshold.

---

## Issue 1 — Người 1

### `[W1][Data] Hoàn thiện data pipeline MVTec AD và khóa protocol chống leakage`

**Workstream:** `1 - Data & Protocol`
**Labels:** `task`, `workstream:data`, `priority:high`
**Assignee:** Người 1
**Reviewer:** Người 4
**Branch gợi ý:** `feat/data/mvtec-loader-split`

#### Mục tiêu

Xây data API dùng chung cho ba category `wood`, `metal_nut`, `capsule`; tạo validation split chỉ từ ảnh normal trong official training set và khóa preprocessing/config để Người 2–4 có thể phát triển model trên cùng một protocol.

#### Deliverables

- Dataset loader trả tối thiểu `image`, `category`, `sample_id`, `source_split` và metadata cần thiết.
- Hỗ trợ ba category `wood`, `metal_nut`, `capsule` với data root truyền qua config hoặc biến môi trường, không hard-code đường dẫn máy cá nhân.
- Deterministic normal validation split và split manifest có `seed`, danh sách sample, category và tỉ lệ split.
- Config preprocessing dùng chung: input resolution, normalization và augmentation nhẹ.
- Hướng dẫn chuẩn bị/cấu trúc dữ liệu local; không commit dataset.
- Sanity tests cho cấu trúc dataset, số lượng mẫu, tính tái lập của split và quy tắc chỉ load mask trong evaluation path.

#### Definition of Done

- [ ] Cả ba category load được và một mini-batch có đúng field/shape đã tài liệu hóa.
- [ ] Cùng config và seed sinh đúng cùng một split manifest.
- [ ] Train và validation chỉ chứa ảnh normal từ official training set.
- [ ] Official test không xuất hiện trong train/validation; ground-truth mask không được đọc ngoài evaluation path.
- [ ] Config và seed được lưu cùng output/split manifest.
- [ ] Test leakage và sanity test chạy xanh trên fixture nhỏ hoặc dataset local.
- [ ] README có lệnh chuẩn bị/chạy kiểm tra dữ liệu.
- [ ] Người 4 review và xác nhận data assumptions cùng quy tắc chống leakage.

#### Dependency / blocker

- Cần mỗi thành viên tự có MVTec AD ở local hoặc một fixture nhỏ hợp lệ để chạy test.
- Đây là dependency chính của Issue 2, 3 và 4; nên ưu tiên chốt public API sớm.

---

## Issue 2 — Người 2

### `[W1][Baseline] Chạy Classical CV và Autoencoder baseline end-to-end trên một category`

**Workstream:** `2 - Baseline Models`
**Labels:** `task`, `workstream:baseline`, `priority:high`
**Assignee:** Người 2
**Reviewer:** Người 5
**Branch gợi ý:** `feat/baseline/classical-ae-pilot`

#### Mục tiêu

Tạo hai baseline chạy được từ config đến prediction chuẩn hóa trên ít nhất một category pilot (mặc định đề xuất `wood`), nhằm có kết quả sơ bộ và dữ liệu đầu vào thật cho pipeline evaluation của Người 5.

#### Deliverables

- Classical CV baseline có config rõ ràng cho preprocessing, Otsu/edge/morphology và cách tạo anomaly map/score.
- Convolutional Autoencoder có entry point train và inference, train chỉ bằng normal train images.
- Quy tắc chuyển reconstruction error thành image-level anomaly score và pixel-level anomaly map.
- Adapter xuất prediction theo `shared/README.md` và lưu artifact/checkpoint ngoài Git.
- Một run sơ bộ cho mỗi baseline trên cùng category pilot, có config snapshot, seed, thời gian inference và đường dẫn artifact.
- Sanity test cho output schema, shape/range anomaly map và khả năng load lại checkpoint.

#### Definition of Done

- [ ] Hai baseline đều chạy end-to-end bằng lệnh/config được tài liệu hóa.
- [ ] Prediction có `experiment_id`, model, category, sample id, anomaly score, anomaly map path, inference time, seed và config path.
- [ ] `anomaly_score` càng cao càng bất thường; anomaly map được resize về kích thước ảnh đánh giá.
- [ ] Không dùng test label/mask để train, chọn hyperparameter hoặc threshold.
- [ ] Cùng checkpoint, ảnh và config cho kết quả tái lập trong sai số đã ghi rõ.
- [ ] Có output mẫu đủ để Người 5 chạy evaluation và Người 6 thử visualization.
- [ ] Checkpoint, dataset và artifact lớn không được commit.
- [ ] Người 5 review output contract, protocol metric và tính tái lập.

#### Dependency / blocker

- Phụ thuộc data API và preprocessing config từ Issue 1.
- Phụ thuộc schema/threshold protocol được Người 5 và Người 6 chốt; trong lúc chờ có thể dùng fixture/dummy adapter theo `shared/README.md`.

---

## Issue 3 — Người 3

### `[W1][PaDiM] Dựng feature extractor và khung fit/predict cho PaDiM`

**Workstream:** `3 - PaDiM`
**Labels:** `task`, `workstream:padim`
**Assignee:** Người 3
**Reviewer:** Người 6
**Branch gợi ý:** `feat/padim/feature-extractor-skeleton`

#### Mục tiêu

Chuẩn bị nền tảng PaDiM cho tuần 2 bằng cách chốt pretrained backbone/layer interface, dựng khung `fit/predict` và xác minh feature tensor trên dữ liệu thật mà chưa yêu cầu hoàn thành Gaussian statistics hay Mahalanobis scoring trong tuần 1.

#### Deliverables

- Module pretrained feature extractor với backbone và layer được điều khiển bằng config.
- Config pilot cho một category, gồm resolution, normalization, backbone, feature layers và seed.
- Skeleton `fit(train_loader, config)` và `predict(image_or_batch, artifact, config)` theo shared interface.
- Smoke test một mini-batch, ghi lại shape từng feature map, kích thước embedding ước tính và hành vi deterministic/eval mode.
- Thiết kế ngắn cho artifact Gaussian statistics: metadata, category, config, feature dimensions và version.
- Unit/sanity test cho output shape, device handling và việc backbone không bị cập nhật ngoài ý muốn.

#### Definition of Done

- [ ] Feature extractor chạy được trên ít nhất một mini-batch từ data API chung.
- [ ] Backbone/layers/resolution không hard-code và có config snapshot.
- [ ] Model ở eval mode; cùng input/config sinh feature nhất quán trong sai số cho phép.
- [ ] Skeleton adapter có chữ ký tương thích shared interface và có test tối thiểu.
- [ ] Không truy cập test label/mask trong feature extraction hoặc fit path.
- [ ] README ghi rõ lệnh smoke test, feature shapes và bước tiếp theo cho tuần 2.
- [ ] Người 6 review khả năng tích hợp adapter và metadata visualization.

#### Dependency / blocker

- Dùng data API/preprocessing từ Issue 1.
- Dùng output contract được khóa trong Issue 6.

---

## Issue 4 — Người 4

### `[W1][PatchCore] Dựng patch feature pipeline và thiết kế memory bank/coreset API`

**Workstream:** `4 - PatchCore`
**Labels:** `task`, `workstream:patchcore`
**Assignee:** Người 4
**Reviewer:** Người 1
**Branch gợi ý:** `feat/patchcore/patch-feature-skeleton`

#### Mục tiêu

Chuẩn bị nền tảng PatchCore cho tuần 2: trích xuất patch-level features, chốt interface memory bank/coreset và đo sơ bộ shape/bộ nhớ trên một mini-batch; chưa yêu cầu hoàn thành nearest-neighbor scoring trong tuần 1.

#### Deliverables

- Module pretrained patch feature extractor với backbone, layers và resolution điều khiển bằng config.
- Hàm chuyển multi-layer feature maps thành patch embeddings, có tài liệu shape vào/ra.
- Interface/skeleton cho memory bank và coreset sampler; `coreset_ratio` là tham số config.
- Smoke test trên một mini-batch, ghi số patch, embedding dimension, RAM/VRAM ước tính và thời gian feature extraction nếu đo được.
- Thiết kế metadata cho memory bank/coreset artifact: experiment id, category, backbone, ratio, seed, config và commit SHA.
- Review Issue 1 về data assumptions, split và leakage theo cặp cross-review.

#### Definition of Done

- [ ] Patch embeddings sinh được từ ít nhất một mini-batch của data API chung.
- [ ] Backbone/layers/resolution/coreset ratio không hard-code.
- [ ] Có test cho feature/embedding shape và input batch size cơ bản.
- [ ] Memory bank/coreset API có type/shape contract rõ để triển khai tiếp ở tuần 2.
- [ ] Không truy cập test label/mask trong feature extraction hoặc fit path.
- [ ] README có lệnh smoke test và số liệu resource sơ bộ.
- [ ] Người 1 review data assumptions; Người 4 hoàn tất review leakage cho Issue 1.

#### Dependency / blocker

- Dùng data API/preprocessing từ Issue 1.
- Dùng output/artifact contract được khóa trong Issue 6.

---

## Issue 5 — Người 5

### `[W1][Evaluation] Khóa metric, threshold protocol và bộ test chuẩn`

**Workstream:** `5 - Evaluation & Research`
**Labels:** `task`, `workstream:evaluation`, `priority:high`
**Assignee:** Người 5
**Reviewer:** Người 2
**Branch gợi ý:** `feat/eval/metric-threshold-protocol`

#### Mục tiêu

Khóa định nghĩa metric và quy tắc threshold dùng chung trước khi so sánh model; tạo evaluator có thể kiểm tra output baseline mà không tune bằng official test labels/masks.

#### Deliverables

- Tài liệu định nghĩa Image AUROC, AP, F1; Pixel AUROC, Dice, IoU và AUPRO/PRO, gồm input, averaging và edge cases.
- Hàm metric cốt lõi và unit tests bằng toy arrays có kết quả biết trước.
- Quy tắc threshold chính lấy từ phân phối normal validation, được điều khiển bằng config và không dùng test label/mask. PR cần chốt rõ một rule mặc định (ví dụ percentile hoặc `mean + k·std`).
- Phân biệt rõ `main validation threshold` với `oracle test threshold`; oracle chỉ được báo như phân tích phụ nếu nhóm quyết định dùng.
- Evaluator/validator đọc shared prediction schema và báo lỗi field/path/shape không hợp lệ.
- Định dạng CSV/JSON kết quả theo model, category, seed và experiment id.

#### Definition of Done

- [ ] Metric cốt lõi khớp expected values trên toy examples và xử lý được trường hợp chỉ có một class bằng cảnh báo rõ ràng.
- [ ] Anomaly map và ground-truth mask được kiểm tra/đưa về cùng kích thước trước pixel metric.
- [ ] Threshold chính chỉ được fit từ normal validation scores và cho kết quả tái lập với cùng config.
- [ ] Test chứng minh evaluation path không làm rò test label/mask vào bước fit threshold.
- [ ] Evaluator đọc được dummy prediction và ít nhất một output mẫu từ Issue 2 khi có.
- [ ] Metric/protocol và result schema được tài liệu hóa đủ để Người 2–4 dùng thống nhất.
- [ ] Người 2 review tính đúng của metric, baseline compatibility và comparison fairness.

#### Dependency / blocker

- Phối hợp Người 1 về validation manifest/config.
- Phối hợp Người 6 để khóa shared result schema; output thật từ Issue 2 chỉ cần cho integration cuối issue.

---

## Issue 6 — Người 6

### `[W1][Integration] Khóa shared contract và chạy dummy prediction qua adapter–visualization`

**Workstream:** `6 - Deployment, Integration & Report`
**Labels:** `task`, `workstream:demo-report`, `priority:high`
**Assignee:** Người 6
**Reviewer:** Người 3
**Branch gợi ý:** `feat/demo/shared-adapter-visualization`

#### Mục tiêu

Khóa interface tích hợp dùng chung và chứng minh một dummy prediction có thể đi từ model adapter qua schema validation đến heatmap/mask/overlay, tạo nền cho demo và báo cáo ở các tuần sau.

#### Deliverables

- PR khóa shared `fit/predict` contract và prediction schema sau khi lấy ý kiến Người 1 và Người 5.
- Model adapter/registry skeleton không phụ thuộc implementation nội bộ của từng model.
- Dummy model hoặc fixture sinh đủ field bắt buộc trong `shared/README.md`.
- Utility render heatmap, binary mask và overlay từ anomaly map; output có đường dẫn tương đối và truy vết experiment id.
- End-to-end integration test: dummy input → adapter → prediction validation → visualization artifacts.
- Tài liệu quy ước artifact directory và cách model owner đăng ký adapter mới.

#### Definition of Done

- [ ] Dummy prediction đi hết pipeline và sinh anomaly score, heatmap, mask, overlay hợp lệ.
- [ ] Schema validator báo lỗi rõ khi thiếu field, sai kiểu hoặc artifact path không tồn tại.
- [ ] Registry gọi model qua interface chung, không `if/else` phụ thuộc chi tiết artifact của từng model trong luồng demo.
- [ ] Threshold trong fixture được truyền từ config/validation protocol, không được tự tune bằng test data.
- [ ] Artifact/figure có experiment id và đường dẫn tương đối; file lớn không commit.
- [ ] Contract được Người 1 và Người 5 đồng thuận; thay đổi sau khi khóa phải qua PR riêng.
- [ ] Người 3 review adapter, output visualization và khả năng tích hợp PaDiM.

#### Dependency / blocker

- Phối hợp Người 1 về data/config metadata và Người 5 về threshold/result schema.
- Không phụ thuộc model hoàn chỉnh; dùng dummy fixture để hoàn thành trong tuần 1.

---

## Gate chung cuối tuần 1

- [ ] Data pipeline chạy được trên `wood`, `metal_nut`, `capsule` và leakage tests xanh.
- [ ] Classical CV và Autoencoder có run sơ bộ trên ít nhất một category.
- [ ] Metric, threshold protocol và shared prediction schema đã được khóa qua review.
- [ ] Dummy prediction đi qua evaluation/visualization thành công.
- [ ] PaDiM và PatchCore có feature extraction skeleton sẵn sàng để hoàn thiện model ở tuần 2.
- [ ] Sáu Pull Request đều liên kết đúng issue, có tối thiểu một approval và CI xanh trước khi merge.
