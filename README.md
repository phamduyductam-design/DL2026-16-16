# Deep Image Inpainting Under Different Missing-Region Patterns

This repository compares a convolutional autoencoder with a small U-Net for reconstructing missing image regions. Both models are trained from scratch on Imagenette. Oxford-IIIT Pet is used only as an external test set. The study varies mask pattern, missing area, and location while keeping the evaluation images and masks paired between models.

The repository contains source code for data preparation, training, evaluation, and an interactive demo. It also includes the locked image selections, the measured single-seed results, and the two best validation-selected model files. The original image datasets and generated mask banks are downloaded or rebuilt separately; see [DATA.md](DATA.md).

## Repository contents

| Path | Purpose |
|---|---|
| [DATA.md](DATA.md) | Official dataset links, versions, splits, preprocessing, and data-reproduction scripts |
| configs/ | Shared dataset, training, and model settings |
| data/manifests/ | Locked lists of selected images and a data audit |
| src/ | Dataset readers, mask generation, models, training, metrics, and inference |
| scripts/ | Commands for data preparation, training, evaluation, plotting, and verification |
| demo/app.py | Interactive comparison of both trained models |
| tests/ | Automated checks for models, masks, splits, losses, and saved results |
| outputs/metrics/ | Curated measured results: the 48-condition summary and 36,000 image-mask evaluations |
| outputs/checkpoints/ | The two best validation-selected model files needed for the demo |
| outputs/logs/ | Training records and experiment identity |
| outputs/figures/ | Selected figures for the main experiments |
| report/results.md | Readable interpretation of the measured results |

## Installation

Python 3.10-3.12 is recommended. From the repository root on Windows PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

On Linux or macOS, activate with source .venv/bin/activate instead. The project can run on CPU, but full training is much faster with a CUDA-capable GPU and a compatible PyTorch build. The recorded experiment used an NVIDIA GeForce RTX 4060 Ti, PyTorch 2.14.1+cu126, batch size 32, and seed 42. Timing numbers depend on hardware and software.

## Data setup

Download the exact official datasets listed in [DATA.md](DATA.md). With the default configs/datasets.yaml, place the Imagenette archive at datasets/imagenette2-160.tgz and extract Oxford-IIIT Pet so that datasets/Oxford-IIIT-Pet/images/ and datasets/Oxford-IIIT-Pet/annotations/test.txt exist. Alternatively, edit the two root paths in that configuration file. The committed manifests are fixed; do not generate a different split for the reported results.

The fixed selections contain 6,000 Imagenette training images, 300 Imagenette validation images, 1,000 Imagenette test images, and 500 Oxford-IIIT Pet test images. Imagenette training and validation come from disjoint selections of the official train partition; the in-domain test comes from the official val partition. The Pet selection comes only from the official test partition.

## Reproduce the main experiment

First run the automated checks and rebuild the fixed validation/test mask banks:

~~~powershell
python -m pytest -q
python scripts/build_mask_bank.py --config configs/base.yaml
~~~

For a complete run, train both models, evaluate both datasets, create figures and tables, and verify all generated artifacts with one command:

~~~powershell
python scripts/run_experiment.py --config configs/base.yaml --seed 42
~~~

The complete run writes to outputs/ and data/mask_bank/ according to configs/base.yaml. It retrains from scratch unless a compatible LAST training state exists. It may replace the included result and best-model files, so use a separate output directory in a copied configuration if you want to retain this measured release unchanged. After the full run, python scripts/check_project.py --config configs/base.yaml --seed 42 returns success only if all required training states, image-level results, banks, figures, and the measured report are consistent. The lightweight repository itself does not contain the large LAST states or mask PNGs, so that command is intended for a completed local run.

The equivalent individual training commands are:

~~~powershell
python scripts/train.py --model autoencoder --config configs/autoencoder.yaml --seed 42
python scripts/train.py --model unet --config configs/unet.yaml --seed 42
~~~

To evaluate the included best model files without retraining, first build the mask banks as above, then run:

~~~powershell
python scripts/evaluate.py --checkpoint outputs/checkpoints/autoencoder_seed42_best.pt --dataset imagenette
python scripts/evaluate.py --checkpoint outputs/checkpoints/autoencoder_seed42_best.pt --dataset pets
python scripts/evaluate.py --checkpoint outputs/checkpoints/unet_seed42_best.pt --dataset imagenette
python scripts/evaluate.py --checkpoint outputs/checkpoints/unet_seed42_best.pt --dataset pets
python scripts/evaluate.py --combine --seed 42
python scripts/plot_results.py
~~~

Checkpoints are selected by Imagenette validation missing-region MAE. Pet results are never used for model selection. The checkpoint and evaluation files record hashes of the locked image lists and mask banks so mismatched data is rejected.

## Recorded results

The following values are averages over the 12 equally weighted mask conditions within each test dataset. Lower MAE-hole is better; higher PSNR-hole and SSIM-full are better. These are measured results from one seed, not estimates of run-to-run uncertainty.

| Test dataset | Model | MAE-hole | PSNR-hole (dB) | SSIM-full | Model-only ms/image |
|---|---|---:|---:|---:|---:|
| Imagenette | Autoencoder | 0.1078 | 16.948 | 0.7910 | 0.797 |
| Imagenette | Small U-Net | 0.1031 | 17.130 | 0.7996 | 0.925 |
| Oxford-IIIT Pet | Autoencoder | 0.1036 | 17.441 | 0.7954 | 0.796 |
| Oxford-IIIT Pet | Small U-Net | 0.0993 | 17.595 | 0.8036 | 0.923 |

The U-Net has lower mean missing-region error in both test datasets, while its measured inference throughput is slower on the recorded GPU. Error rises as more image area is hidden. The [full 48-condition table](outputs/metrics/summary.csv), [image-level records](outputs/metrics/per_image.csv), and [result discussion](report/results.md) preserve the pattern- and size-specific results. The image-level file contains 36,000 evaluations: 1,000 Imagenette and 500 Pet images, each under 12 conditions, for both models. It is included once rather than as duplicate per-model exports.

The experiment was produced from source commit 6a5a612b64813bee96bb9e9ed48f53f15fd13595. The included training logs identify the same commit and the same locked data. The results were checked against the 36,000 image-level records, and both included best model files load into the current model definitions. These checks do not replace an independent full rerun.

## Interactive demo

~~~powershell
python demo/app.py
~~~

The demo uses the included best model files by default. It takes an uploaded image as a reference, hides a synthetic region, and compares reconstructions of the same image and mask. Its MAE, PSNR, and SSIM values measure similarity to that reference; they are not confidence scores and do not establish performance on an image whose original content is unknown.

## Experiment design and limits

The model input has four channels: masked RGB plus a binary mask. Both models predict RGB; known pixels are copied from the input image in the restored output. Training minimizes missing-region MAE plus 0.1 times known-region MAE. The four patterns are center rectangle, random rectangle, free-form strokes, and six disconnected holes at target missing areas of 15%, 30%, and 50%. Pattern comparisons use masks matched in actual missing area within one percentage point per image and target area.

Images are evaluated at 128 x 128 pixels. The main comparison uses one training seed and compact models, so small numerical differences should not be interpreted as statistical significance. Full-image scores benefit from unchanged known pixels; missing-region scores are more direct measures of inpainting quality.
