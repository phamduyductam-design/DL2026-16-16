# Four-slide project overview (maximum three minutes)

## Slide 1 — Problem and research question

- Reconstruct missing RGB regions from visible image context.
- Ask how mask size, mask shape, and mask position change reconstruction quality, and whether a small U-Net improves on a convolutional autoencoder.
- Show one reference image, its masked input, and both reconstructions.

## Slide 2 — Method and experiments

- Train both models from scratch on the same Imagenette selection: 6,000 training and 300 validation images.
- Evaluate on 1,000 Imagenette test images and 500 Oxford-IIIT Pet official-test images; Pet is never used for training or checkpoint selection.
- Compare 12 paired conditions: four mask patterns at 15%, 30%, and 50% target missing area. Select the best saved model using Imagenette validation MAE-hole.
- Report missing-region error, PSNR, whole-image SSIM, and model-only inference throughput.

## Slide 3 — Key results

- Mean Imagenette MAE-hole: autoencoder 0.1078, small U-Net 0.1031.
- Mean Pet MAE-hole: autoencoder 0.1036, small U-Net 0.0993.
- Missing-region error increases as the hidden area grows. Center rectangles are slightly harder than paired random rectangles in this experiment.
- Show the model-comparison figure and one size or pattern figure from outputs/figures/. State clearly that this is one training seed.

## Slide 4 — Conclusion and demo

- The small U-Net reduces mean missing-region error in both test sets, with a modestly slower measured model-only throughput.
- The selected Pet test set does not show worse mean error than Imagenette, so avoid claiming a universal domain-transfer failure.
- Demonstrate a single uploaded reference image with one fixed synthetic mask and switch between the two models.
- Note the limits: 128 x 128 images, compact models, one seed, and metrics measured against a known reference rather than confidence.

Use report/results.md and outputs/metrics/summary.csv for exact values. Keep the spoken overview under three minutes; examination questions follow separately.
