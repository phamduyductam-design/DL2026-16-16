import time
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models import make_model
from src.datasets import InpaintingDataset
from src.utils.config import (
    load_config,
    project_path,
    bank_path,
    experiment_identity,
    file_sha256
)
from src.utils.checkpoint import load_checkpoint, model_state_sha256
from src.training.engine import get_device
from src.training.losses import model_input
from .metrics import metrics


def evaluate(checkpoint_path, dataset, dataset_config=None, batch_size=32):

    checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint.get("checkpoint_kind") != "best":
        raise ValueError(
            "Final evaluation requires an Imagenette-validation BEST checkpoint"
        )

    config = checkpoint["config"]

    if dataset_config is None:
        dataset_config = config["dataset_config"]

    live_config = dict(config, dataset_config=dataset_config)
    identity = experiment_identity(live_config)

    if checkpoint.get("data_identity") != identity:
        raise ValueError(
            "Evaluation manifests/banks differ from the checkpoint experiment"
        )

    saved_model_hash = checkpoint.get("model_state_sha256")
    current_model_hash = model_state_sha256(
        checkpoint["model_state_dict"]
    )

    if current_model_hash != saved_model_hash:
        raise ValueError("BEST model content hash mismatch")

    checkpoint_hash = file_sha256(checkpoint_path)

    model_name = config["model"]
    seed = config["train_seed"]

    device = get_device(config)

    model = make_model(model_name)
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset_settings = load_config(dataset_config)

    test_manifest = (
        project_path(dataset_settings["manifest_dir"])
        / f"{dataset}_test.csv"
    )

    mask_bank = bank_path(
        config,
        dataset,
        "test"
    )

    test_data = InpaintingDataset(
        dataset_settings[dataset]["root"],
        test_manifest,
        bank=mask_bank
    )

    loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    rows = []

    with torch.inference_mode():

        first_batch = next(iter(loader))

        first_image = first_batch["image"].to(device)
        first_mask = first_batch["mask"].to(device)

        warmup_input = model_input(
            first_image,
            first_mask
        )

        model(warmup_input)

        for batch in loader:

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            inputs = model_input(
                images,
                masks
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            predictions = model(inputs)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed_time = time.perf_counter() - start_time

            inference_ms = (
                elapsed_time / len(images)
            ) * 1000

            batch_metrics = metrics(
                predictions,
                images,
                masks
            )

            parameter_count = sum(
                p.numel()
                for p in model.parameters()
            )

            for i in range(len(images)):

                condition = batch["condition"][i]

                row = {
                    "train_dataset": "imagenette",
                    "eval_dataset": dataset,
                    "model": model_name,
                    "train_seed": seed,

                    "image_id": batch["image_id"][i],
                    "mask_id": batch["mask_id"][i],

                    "pattern": condition[0],
                    "size": int(condition[1:]),
                    "condition": condition,

                    "actual_ratio":
                        float(batch["actual_ratio"][i]),

                    "inference_ms": inference_ms,
                    "parameter_count": parameter_count,
                    "timing_batch_size": batch_size,

                    "checkpoint_sha256": checkpoint_hash,

                    "manifest_sha256":
                        identity["manifests"][
                            f"{dataset}_test"
                        ],

                    "bank_sha256":
                        identity["banks"][
                            f"{dataset}_test"
                        ]["bank_sha256"]
                }

                for metric_name, values in batch_metrics.items():
                    row[metric_name] = float(values[i])

                rows.append(row)

            if len(rows) % 1200 == 0:
                print(
                    f"{model_name}/{dataset}: "
                    f"{len(rows)}/{len(test_data)}",
                    flush=True
                )

    output_dir = (
        project_path(config["output_dir"])
        / "metrics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    frame = pd.DataFrame(rows)

    if dataset == "imagenette":
        expected = 12000
    else:
        expected = 6000

    if (
        len(frame) != expected
        or frame["mask_id"].nunique() != expected
    ):
        raise ValueError(
            f"Wrong pair count: "
            f"{len(frame)}, expected {expected}"
        )

    per_image_file = (
        output_dir
        / f"{model_name}_seed{seed}_{dataset}_per_image.csv"
    )

    frame.to_csv(
        per_image_file,
        index=False
    )

    summary = aggregate(frame)

    summary_file = (
        output_dir
        / f"{model_name}_seed{seed}_{dataset}_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print(
        f"Saved {per_image_file}",
        flush=True
    )

    return frame


def aggregate(frame):

    group_columns = [
        "train_dataset",
        "eval_dataset",
        "model",
        "train_seed",
        "pattern",
        "size",
        "condition"
    ]

    metric_columns = [
        "actual_ratio",
        "MAE_hole",
        "MSE_hole",
        "PSNR_hole",
        "PSNR_full",
        "SSIM_full",
        "inference_ms",
        "parameter_count"
    ]

    grouped = frame.groupby(
        group_columns,
        sort=True
    )

    summary = (
        grouped[metric_columns]
        .mean()
        .reset_index()
    )

    summary["n_pairs"] = (
        grouped.size().to_numpy()
    )

    return summary


def combine_results(output="outputs", seed=42):

    metrics_dir = (
        project_path(output)
        / "metrics"
    )

    files = []

    models = [
        "autoencoder",
        "unet"
    ]

    datasets = [
        "imagenette",
        "pets"
    ]

    for model in models:
        for dataset in datasets:

            file_path = (
                metrics_dir
                / f"{model}_seed{seed}_{dataset}_per_image.csv"
            )

            files.append(file_path)

    if not all(file.exists() for file in files):
        raise FileNotFoundError(
            "Need all four model/dataset evaluations before combining"
        )

    frames = [
        pd.read_csv(file)
        for file in files
    ]

    autoencoder_frames = frames[:2]
    unet_frames = frames[2:]

    paired_columns = [
        "image_id",
        "mask_id",
        "condition",
        "actual_ratio",
        "manifest_sha256",
        "bank_sha256",
        "timing_batch_size"
    ]

    for autoencoder_frame, unet_frame in zip(
        autoencoder_frames,
        unet_frames
    ):

        if not autoencoder_frame[paired_columns].equals(
            unet_frame[paired_columns]
        ):
            raise ValueError(
                "Models were evaluated on different image/mask pairs"
            )

    for frame in frames:

        areas = (
            frame
            .groupby(["image_id", "size"])
            .actual_ratio
            .agg(["min", "max"])
        )

        area_difference = (
            areas["max"]
            - areas["min"]
        )

        if (area_difference > 0.01).any():
            raise ValueError(
                "Evaluation patterns have unmatched areas"
            )

    for model_frames in [
        autoencoder_frames,
        unet_frames
    ]:

        checkpoint_hashes = set()

        for frame in model_frames:
            checkpoint_hashes.update(
                frame.checkpoint_sha256.unique()
            )

        if len(checkpoint_hashes) != 1:
            raise ValueError(
                "Domain evaluations used different checkpoints"
            )

    combined_frame = pd.concat(
        frames,
        ignore_index=True
    )

    summary = aggregate(
        combined_frame
    )

    if (
        len(summary) != 48
        or len(combined_frame) != 36000
    ):
        raise ValueError(
            "Expected 48 summary rows "
            "and 36000 pairs per seed"
        )

    combined_frame.to_csv(
        metrics_dir / "per_image.csv",
        index=False
    )

    summary.to_csv(
        metrics_dir / "summary.csv",
        index=False
    )

    combined_frame.to_csv(
        metrics_dir / f"per_image_seed{seed}.csv",
        index=False
    )

    summary.to_csv(
        metrics_dir / f"summary_seed{seed}.csv",
        index=False
    )

    print(
        "Validated 48 rows/seed "
        "and 36000 paired evaluations"
    )

    return summary
