# Presentation material (12 slides)

1. **Topic** — Deep Image Inpainting; show GT → mask → restored.
2. **Research questions** — pattern, size, position, AE vs U-Net, Pet generalization.
3. **Datasets** — V2 Imagenette 6000/300 from disjoint official train selections, 1000 from official val; Pet 500 official test, 13–14 per breed; SHA256 and approximate near-duplicate screening.
4. **12 conditions** — C/R/F/H × 15/30/50%; fixed banks, paired rectangles, maximum between-pattern area spread ≤0.98 percentage points per image/ratio.
5. **Autoencoder** — 4-channel input, encoder 32/64/128, bottleneck 256@16, decoder 128/64/32, no skips, sigmoid.
6. **Small U-Net** — same encoder with skips at 128, 64, 32; bilinear upsampling.
7. **Loss/training/metrics** — per-image hole MAE + 0.1 valid MAE; Adam, validation-only selection; six metrics including inference time.
8. **Model comparison** — To be measured. Use real E1 charts and results.md; include parameter count and timing setup.
9. **Mask size/pattern/position** — To be measured. Use E2–E4 charts with held-constant factors and paired C/R.
10. **Pet generalization/failures** — To be measured. Use E6 chart and ranked failure grids; no Pet fine-tuning or unproven claims.
11. **Gradio demo** — uploaded reference image, synthetic fixed mask, switch AE/U-Net; PSNR/SSIM measure reference similarity, not confidence or proof for already damaged images.
12. **Conclusion/limitations** — Pending experiment. Answer questions from real data; discuss 128px images, model capacity, single-seed limits.

Speaker note: slides 8–10 must reference outputs/metrics/summary.csv and generated figures. Never substitute example numbers for measured results.
