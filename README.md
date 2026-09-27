# Deep Image Inpainting — bản nộp mã nguồn

So sánh Convolutional Autoencoder và Small U-Net train từ đầu trên Imagenette, với 12 condition: Center Rectangle / Random Rectangle / Free-form / Multiple Holes × 15% / 30% / 50%. Oxford-IIIT Pet chỉ dùng đánh giá, không train hoặc fine-tune.

## Nội dung

- `src/`: dataset, mask, hai model, loss, training, metrics và inference.
- `scripts/`: các lệnh chuẩn bị dữ liệu, train, evaluate và tạo biểu đồ.
- `configs/`: cấu hình dataset và thí nghiệm.
- `demo/app.py`: demo Gradio.
- `tests/` và `pytest.ini`: kiểm tra tính đúng của code.
- `data/manifests/`: danh sách train/validation/test cố định và audit dữ liệu.
- `report/`: dàn ý báo cáo, nội dung slide và tài liệu tham khảo; kết quả chưa đo được để Pending experiment.
- `requirements.txt`: thư viện cần cài.

Bản nộp không kèm dataset, môi trường Python, mask PNG, checkpoint hoặc log. Mask được tạo lại đúng bằng seed cố định; checkpoint và kết quả được sinh khi chạy. Đây là bản mã nguồn, không phải gói kết quả thí nghiệm đã hoàn tất.

## Cài đặt

Mở terminal tại thư mục chứa README này. Dùng Python Windows chính thức 3.10–3.12 hoặc Python trong Colab/Linux:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/Colab dùng `source .venv/bin/activate` thay cho lệnh activate của Windows. PyTorch hỗ trợ CPU; để train trên GPU, cài bản CUDA tương thích máy. Thiết lập đã kiểm tra: PyTorch 2.6.0 + CUDA 12.4, RTX 4060; lệnh tương ứng:

```powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## Dataset

Sửa `configs/datasets.yaml` để trỏ đến dataset **đã có**. Imagenette nhận file `.tgz` hoặc thư mục có `train/`, `val/`; Pet cần `images/` và `annotations/test.txt`. Không có bước tải lại hoặc copy dataset. Đường dẫn tương đối tính từ thư mục project.

Manifest đã kèm bản nộp: Imagenette 6000 train / 300 validation / 1000 test, Pet 500 official test; Imagenette cân bằng theo class. Hash nội dung được kiểm tra khi load. Nếu dùng cùng dataset, giữ nguyên manifest để tái lập split.

Protocol V2: train 600/class và validation 30/class là hai phần rời nhau từ **official train**; test 100/class từ **official val**. Pet chọn gần cân bằng: 19 breed × 14 ảnh + 18 breed × 13 ảnh, tất cả từ official test. Đã sàng lọc trùng SHA256 và gần trùng trước khi chia dữ liệu. Gần trùng dùng pHash Hamming ≤4, sau đó xác nhận thumbnail RGB 32×32 có MAE ≤0.04 và MSE ≤0.003. Đây là sàng lọc xấp xỉ, không chứng minh mọi gần trùng ngữ nghĩa đều được loại. Audit ghi chính sách, các cặp loại bỏ và SHA256 từng manifest.

Manifest và mask generator đã đổi sang V2. **Checkpoint/bank của bản cũ không dùng lại được cho thí nghiệm này; cần train lại.** Không trộn kết quả hai protocol. Đường dẫn dataset trong YAML là mẫu để sửa; không phải dataset đã được đóng gói.

`scripts/prepare_data.py` dùng khi chuẩn bị một thí nghiệm mới, không ghi đè manifest đã khóa. Để chạy lệnh này với dữ liệu khác, đặt `manifest_dir` thành thư mục mới trong YAML:

```powershell
python scripts/prepare_data.py --config configs/datasets.yaml
```

## Chạy project

```powershell
python -m pytest -q
python scripts/build_mask_bank.py --config configs/base.yaml
python scripts/train.py --model autoencoder --config configs/autoencoder.yaml --seed 42
python scripts/train.py --model unet --config configs/unet.yaml --seed 42
```

Train tự kiểm tra overfit 16 ảnh với mask cố định trước khi train đầy đủ; fail thì dừng. Hai model dùng cùng preprocessing, thứ tự batch, ảnh/mask và batch size. Nếu thiếu VRAM, đổi batch size từ 32 sang 16 cho cả hai model và dùng output_dir mới. CPU có thể test bằng `device: cpu` trong base.yaml.

Resume bằng đúng cấu hình cũ:

```powershell
python scripts/train.py --model autoencoder --config configs/autoencoder.yaml --seed 42 --resume outputs/checkpoints/autoencoder_seed42_last.pt
```

BEST chọn theo Imagenette validation MAE-hole; LAST chứa optimizer, scheduler và RNG để resume. Không dùng Pet chọn checkpoint.

BEST được lưu trước LAST. LAST còn chứa snapshot BEST để khi resume có thể sửa BEST thiếu, cũ hoặc mới hơn LAST do bị ngắt giữa hai lần lưu. Cả checkpoint và bảng evaluation gắn hash manifest/bank. Bank đã có vẫn phải kiểm tra PNG, metadata, seed, version và tái sinh đối chiếu; không giữ bank chỉ vì có metadata.csv.

```powershell
python scripts/evaluate.py --checkpoint outputs/checkpoints/autoencoder_seed42_best.pt --dataset imagenette
python scripts/evaluate.py --checkpoint outputs/checkpoints/autoencoder_seed42_best.pt --dataset pets
python scripts/evaluate.py --checkpoint outputs/checkpoints/unet_seed42_best.pt --dataset imagenette
python scripts/evaluate.py --checkpoint outputs/checkpoints/unet_seed42_best.pt --dataset pets
python scripts/evaluate.py --combine --seed 42
python scripts/plot_results.py
python demo/app.py
```

Hoặc chạy toàn pipeline tuần tự:

```powershell
python scripts/run_experiment.py --seed 42
```

Pipeline đọc `output_dir` và `mask_bank_dir` từ cùng config cho cả hai model; hỗ trợ config riêng bằng `python scripts/run_experiment.py --config configs/base.yaml --seed 42`. Các lệnh combine/plot độc lập nhận `--output`; plot nhận thêm `--mask-bank-dir`. Khi đổi cấu hình nghiên cứu, chọn output_dir và mask_bank_dir mới.

`python scripts/check_project.py --config configs/base.yaml --seed 42` kiểm tra dữ liệu khóa, checkpoint, run hoàn tất, metrics và sản phẩm bắt buộc. Lệnh trả exit code **1** nếu thiếu/sai sản phẩm, **0** chỉ khi tất cả kiểm tra đạt. Kiểm tra này cần dataset thật tại các đường dẫn YAML.

`outputs/` và `data/mask_bank/` sẽ tự được tạo. Kết quả đầy đủ gồm 36000 image-mask pairs và summary 48 dòng/seed, tách riêng hai dataset. Figures, failure cases và bảng báo cáo chỉ dùng số đo thật.

## Thiết kế chính

Input 4 kênh: ảnh che vùng thiếu + binary mask; output RGB sigmoid. Pixel known của restored giữ nguyên ground truth. Loss tính per image: MAE-hole + 0.1 MAE-valid. Metrics: MAE-hole, MSE-hole, PSNR-hole, PSNR-full, SSIM-full và inference time. C/R dùng cùng rectangle size, chỉ khác vị trí; H có đúng 6 components; Free-form có attempt cap.

Generator V2 dùng diện tích rectangle C/R làm tham chiếu cho F/H, rejection sampling trong ±0.49 điểm phần trăm quanh tham chiếu. Vì vậy độ chênh lớn nhất giữa bốn pattern trên cùng ảnh/mức thiếu ≤0.98 điểm phần trăm; đồng thời mỗi mask vẫn đạt khoảng target ±2 điểm phần trăm. E3 kiểm tra ghép diện tích trước khi xuất kết quả.

Demo coi ảnh upload là ground truth rồi che nhân tạo. PSNR/SSIM/MAE đo độ giống ảnh gốc này, **không phải confidence**, và không chứng minh phục hồi đúng ảnh đã hỏng sẵn khi không có ground truth.

Adam LR=0.001, tối đa 30 epoch, early stopping 5. ReduceLROnPlateau factor=0.5, patience=1 trong API để giảm sau hai epoch không cải thiện. Inference time là model-only throughput ms/image sau warm-up, synchronize CUDA, không phải latency batch=1.
