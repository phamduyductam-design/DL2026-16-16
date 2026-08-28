# Experiment Configs

Owner chính: **Người 1 - Data & Protocol Lead**. Đồng review: **Người 5**.

Mỗi experiment cần lưu tối thiểu:

- `experiment_id`, model, category và seed;
- input resolution, normalization và augmentation;
- backbone và layer/features sử dụng;
- tham số train hoặc fit;
- threshold rule lấy từ normal validation;
- đường dẫn output/artifact tương đối;
- phiên bản code hoặc commit SHA khi chạy kết quả chính.

Tên file đề xuất:

```text
<model>_<category>_<variant>.yaml
```

Ví dụ: `patchcore_wood_resnet18_r256_c01.yaml`.

