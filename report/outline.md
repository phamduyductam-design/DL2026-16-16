# Deep Image Inpainting under Different Missing-Region Patterns

## 1 Introduction
Reconstruct missing RGB pixels from visible context. Study effects of pattern, area, position and model skip connections.

## 2 Problem Definition
Images 128×128, binary M=1 at missing pixels. Input concat(I(1−M), M); output P in [0,1]. Restored image PM + I(1−M).

## 3 Related Work
Discuss encoder-decoder reconstruction, U-Net skip connections, free-form masks and image quality metrics. Partial convolution / GAN methods are related work only, not implemented baselines. See references.bib.

## 4 Dataset
Protocol V2. Imagenette: 6000/300/1000, balanced by class; train and validation are disjoint selections from official train, test from official val. Pet: fixed 500 official test images, 19 breeds with 14 images and 18 with 13, evaluation only. No tuning/fine-tuning on Pet. SHA256 deduplication plus conservative pHash candidate screening and RGB thumbnail confirmation precede splitting. Approximate near-duplicate screening is not a guarantee against every semantic duplicate. Audit locks the protocol and manifest hashes.

## 5 Mask Generation
Generator V2: C/R/F/H × 15/30/50%; accepted area ±2 percentage points. C/R paired geometry. F/H area is matched to the C/R reference within ±0.49 percentage points: maximum between-pattern spread ≤0.98 percentage points per image/ratio. Free-form bounded rejection sampling. Six disconnected H holes. Fixed validation/test bank with seed/version/attempt count and content fingerprints. Present mask_examples.png and mask_area_histogram.png.

## 6 Methodology
Two scratch models: AE without skips, Small U-Net with 128/64/32-resolution skips. Shared encoder channels 32/64/128, bottleneck 256 at 16×16. Bilinear upsampling with convolution. Per-image hole MAE + 0.1 valid MAE.

## 7 Experimental Setup
Adam LR=.001, batch=32 (or shared 16), 30 maximum epochs, early stop=5, main seed=42. Best checkpoint selected only using Imagenette validation MAE-hole. Scheduler halves LR after two failed validation epochs. Metrics are computed per image then averaged per condition. Model-only synchronized inference throughput is reported with batch size.

LAST embeds the BEST snapshot for transaction recovery; resume/evaluation reject changed manifest or mask bank fingerprints. This V2 experiment requires new training; results/checkpoints from the previous protocol are not interchangeable.

## 8 Results
The measured 48-condition summary and 36,000 image-level evaluations are in outputs/metrics/. See [results.md](results.md) for the model comparison and interpretation. E1 compares models; E2 examines size; E3 compares patterns; E4 uses paired rectangle positions; E6 compares the two test datasets. Parameter counts and inference throughput are reported with the recorded GPU and batch-size context.

## 9 Discussion
The small U-Net has lower mean missing-region error on both selected test sets. Error grows with missing area. Pet has lower mean error than Imagenette on these selections, so the measured results do not establish that the external dataset is harder. Pattern effects are specific to these masks and images. One seed does not establish statistical significance.

## 10 Failure Cases
The complete run produced good, median, and bad examples ranked by paired mean MAE-hole per condition and dataset. The image-level records and plotting code can regenerate the grids. Discuss texture, object structure, and boundary errors using the selected image and mask IDs. Full-image metrics benefit from unchanged known pixels.

## 11 Conclusion
Answer the research questions from [results.md](results.md). Acknowledge the 128 x 128 resolution, compact scratch models, single main seed, and selected dataset subsets.
