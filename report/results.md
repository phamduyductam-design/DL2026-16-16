# Measured results

These results come from one complete seed-42 experiment. The two models were trained on the same 6,000 Imagenette images and selected by the same 300-image Imagenette validation split. Each model was evaluated on 1,000 held-out Imagenette images and 500 Oxford-IIIT Pet test images under all 12 mask conditions. The [condition-level summary](../outputs/metrics/summary.csv) has 48 rows; the [image-level table](../outputs/metrics/per_image.csv) has 36,000 rows.

## Main model comparison

The table gives the equally weighted mean of the 12 conditions within each test dataset. MAE-hole measures average absolute error only in the missing region. PSNR-hole is also restricted to the missing region; SSIM-full uses the entire restored image.

| Test dataset | Model | MAE-hole | PSNR-hole (dB) | SSIM-full | Parameters | Model-only ms/image |
|---|---|---:|---:|---:|---:|---:|
| Imagenette | Autoencoder | 0.10778 | 16.948 | 0.7910 | 1,947,459 | 0.797 |
| Imagenette | Small U-Net | 0.10313 | 17.130 | 0.7996 | 2,140,995 | 0.925 |
| Oxford-IIIT Pet | Autoencoder | 0.10360 | 17.441 | 0.7954 | 1,947,459 | 0.796 |
| Oxford-IIIT Pet | Small U-Net | 0.09934 | 17.595 | 0.8036 | 2,140,995 | 0.923 |

The U-Net has a lower mean missing-region error on both datasets: by 0.00465 on Imagenette and 0.00426 on Pet. It also has more parameters and a slower recorded model-only inference throughput. This is a measured single-seed comparison, not evidence of statistical significance. See the [Imagenette model comparison](../outputs/figures/imagenette_model_MAE_hole.png) and [Pet model comparison](../outputs/figures/pets_model_MAE_hole.png).

## Missing area, pattern, and position

Across all four patterns on Imagenette, autoencoder MAE-hole rises from 0.09515 at 15% missing area to 0.12068 at 50%; U-Net rises from 0.08921 to 0.11711. The same direction appears on Pet. See the [mask-size figure](../outputs/figures/imagenette_size_MAE_hole.png).

For these data and this mask generator, rectangular masks have higher average missing-region error than the free-form and multiple-hole masks at the corresponding target areas. The comparison controls actual masked area within one percentage point for each image and target area, but does not make the shapes identical. See the [pattern figure](../outputs/figures/imagenette_pattern.png) and [mask examples](../outputs/figures/mask_examples.png).

Random rectangles produce slightly lower mean missing-region error than center rectangles in both models and datasets. Center and random rectangles share dimensions for each image, so this is a paired position comparison. See the [position figure](../outputs/figures/imagenette_position.png).

## External test set and interpretation

For the same saved model files, mean missing-region error on the selected Pet test images is lower than on the selected Imagenette test images. Therefore these measurements do not support a blanket claim that the Pet domain is harder. Image content and the selected test subsets can affect the comparison. See the [U-Net domain figure](../outputs/figures/unet_domain_comparison.png). Pet was not used for training or model selection.

The full local run also generated good, median, and bad reconstruction grids selected by paired mean missing-region error. Their selection identifiers are reproducible from the image-level table and the plotting script. A failed reconstruction can blur or invent structured content in a large hidden region; the unchanged visible pixels can make whole-image metrics look better than missing-region metrics. The compact repository keeps the core numeric records and selected figures; the full grids can be regenerated from the official datasets and model files.

## Reproduction and limits

The experiment used source commit 6a5a612b64813bee96bb9e9ed48f53f15fd13595, training seed 42, mask seed 314159, batch size 32, PyTorch 2.14.1+cu126, and an NVIDIA GeForce RTX 4060 Ti. Both models completed 30 epochs, and their best Imagenette-validation scores occurred at epoch 30. The included training records preserve the configuration and dataset fingerprints. The 48-row summary was recomputed from the 36,000 image-level records and matched. Both saved best models load into the current definitions and produce a 128 x 128 RGB output.

Only one training seed was run. The 128 x 128 resolution, compact models, selected dataset subsets, approximate near-duplicate screening, and hardware-dependent timing limit generalization. See [DATA.md](../DATA.md) for exact source versions, splits, and preprocessing, and [README.md](../README.md) for reproduction commands.
