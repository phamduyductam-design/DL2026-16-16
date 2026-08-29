# Dataset source and usage note

## Selected scope

Week 1 uses only these MVTec AD categories:

- `wood`
- `metal_nut`
- `capsule`

`cable` remains an optional extension and is not downloaded for the week-1 gate.

## Source records

- Kaggle mirror: <https://www.kaggle.com/datasets/nghiavutrong/dataset-deep-learning-mvtec>
- Kaggle handle: `nghiavutrong/dataset-deep-learning-mvtec`
- Retrieved: `2026-08-29`
- Full mirror shown by Kaggle: approximately 5.27 GB and 6,645 files.
- Selective paths downloaded: `MVTecAD/wood`, `MVTecAD/metal_nut`, `MVTecAD/capsule`.
- The Kaggle Data Card has no description and displays its license as `Unknown`.

The downloaded category folders contain the original `readme.txt` and
`license.txt`. Those files identify the data as MVTec AD, request citation of
the CVPR 2019 paper, and specify CC BY-NC-SA 4.0. This agrees with the official
MVTec page:

- Official dataset page: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad>
- Official paper: <https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/05_research_teaching/datasets/mvtec_ad.pdf>
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International (CC BY-NC-SA 4.0); commercial use is not permitted by the
  standard license.

## Required citation

Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger,
“MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly
Detection,” IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2019, pp. 9592–9600, DOI: 10.1109/CVPR.2019.00982.

## Provenance warning

The project received the files from a third-party Kaggle mirror, not directly
from the official MVTec download form. Do not claim that the mirror is an
official redistribution. Keep the embedded readme/license files, record the
source URL and run the repository structure/decoding/hash audits before using
the files in an experiment.
