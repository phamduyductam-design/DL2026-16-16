import csv
import json
import os
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import InpaintingDataset
from src.models import make_model

from src.utils.config import (
    ROOT,
    load_config,
    project_path,
    bank_path,
    experiment_identity
)

from src.utils.seed import (
    seed_everything,
    random_states,
    restore_states,
    stable_seed
)

from src.utils.checkpoint import (
    load_checkpoint,
    commit_training_checkpoints,
    repair_best_checkpoint
)

from src.evaluation.metrics import tensor_metrics

from .losses import (
    inpainting_loss,
    model_input,
    restore
)


def get_device(config):
    requested_device = config.get("device", "auto")

    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    return torch.device(requested_device)


def data_for(config, split, train=False):
    dataset_config = load_config(
        config["dataset_config"]
    )

    image_root = dataset_config["imagenette"]["root"]

    manifest_file = (
        project_path(
            dataset_config["manifest_dir"]
        )
        / f"imagenette_{split}.csv"
    )

    if train:
        mask_bank = None
    else:
        mask_bank = bank_path(
            config,
            "imagenette",
            split
        )

    dataset = InpaintingDataset(
        image_root,
        manifest_file,
        train=train,
        bank=mask_bank,
        train_seed=config["train_seed"],
        mask_seed=config["mask_seed"]
    )

    return dataset


def sanity_check(config, model_name, steps=80):
    seed_everything(
        config["train_seed"]
    )

    device = get_device(config)

    model = make_model(
        model_name
    ).to(device)

    dataset = data_for(
        config,
        "train",
        train=True
    )

    dataset.train = False

    subset = torch.utils.data.Subset(
        dataset,
        range(16)
    )

    loader = DataLoader(
        subset,
        batch_size=16
    )

    batch = next(
        iter(loader)
    )

    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"]
    )

    model.eval()

    with torch.no_grad():
        inputs = model_input(
            images,
            masks
        )

        predictions = model(
            inputs
        )

        initial_loss = float(
            inpainting_loss(
                predictions,
                images,
                masks
            )
        )

    model.train()

    history = []

    for step in range(steps):
        optimizer.zero_grad(
            set_to_none=True
        )

        inputs = model_input(
            images,
            masks
        )

        predictions = model(
            inputs
        )

        assert predictions.shape == images.shape
        assert predictions.min() >= 0
        assert predictions.max() <= 1

        loss = inpainting_loss(
            predictions,
            images,
            masks
        )

        loss.backward()

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Sanity check: non-finite loss"
            )

        gradients_are_valid = all(
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        )

        if not gradients_are_valid:
            raise RuntimeError(
                "Sanity check: non-finite gradients"
            )

        optimizer.step()

        current_loss = float(
            loss.detach()
        )

        history.append(
            current_loss
        )

        if step % 10 == 0:
            print(
                f"Sanity {model_name}: "
                f"{step}/{steps}, "
                f"loss={current_loss:.5f}",
                flush=True
            )

    model.eval()

    with torch.no_grad():
        inputs = model_input(
            images,
            masks
        )

        predictions = model(
            inputs
        )

        final_loss = float(
            inpainting_loss(
                predictions,
                images,
                masks
            )
        )

        restored = restore(
            predictions,
            images,
            masks
        )

        assert torch.equal(
            restored * (1 - masks),
            images * (1 - masks)
        )

    required_loss = (
        0.85 * initial_loss
    )

    if final_loss >= required_loss:
        raise RuntimeError(
            "Overfit did not reduce loss >=15%: "
            f"{initial_loss:.5f} -> "
            f"{final_loss:.5f}"
        )

    result = {
        "model": model_name,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "steps": steps,
        "images": 16,
        "passed": True,
        "history": history
    }

    log_dir = (
        project_path(
            config["output_dir"]
        )
        / "logs"
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    sanity_file = (
        log_dir
        / f"{model_name}_sanity.json"
    )

    sanity_file.write_text(
        json.dumps(
            result,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        f"Sanity PASSED: "
        f"{initial_loss:.5f} -> "
        f"{final_loss:.5f}",
        flush=True
    )

    return result


def train(config, model_name, resume=None):
    config = dict(
        config,
        model=model_name
    )

    identity = experiment_identity(
        config
    )

    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        ":4096:8"
    )

    seed_everything(
        config["train_seed"]
    )

    device = get_device(
        config
    )

    if resume is None:
        sanity_check(
            config,
            model_name
        )

        seed_everything(
            config["train_seed"]
        )

    model = make_model(
        model_name
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=1,
        threshold=0
    )

    start_epoch = 0
    best_metric = float("inf")
    best_epoch = 0
    bad_epochs = 0

    output_dir = project_path(
        config["output_dir"]
    )

    run_id = (
        f"{model_name}_seed"
        f"{config['train_seed']}"
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    best_path = (
        checkpoint_dir
        / f"{run_id}_best.pt"
    )

    best_snapshot = None

    if resume is not None:
        checkpoint = load_checkpoint(
            resume,
            device
        )

        if checkpoint["config"] != config:
            raise ValueError(
                "Resume config differs; "
                "use the saved config exactly"
            )

        if checkpoint.get(
            "data_identity"
        ) != identity:
            raise ValueError(
                "Resume data changed: "
                "manifest/bank fingerprints differ "
                "or are missing"
            )

        best_snapshot = repair_best_checkpoint(
            checkpoint,
            best_path
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

        start_epoch = checkpoint[
            "epoch"
        ]

        best_metric = checkpoint[
            "best_metric"
        ]

        best_epoch = checkpoint[
            "best_epoch"
        ]

        bad_epochs = checkpoint[
            "bad_epochs"
        ]

        restore_states(
            checkpoint[
                "random_states"
            ]
        )

    train_data = data_for(
        config,
        "train",
        train=True
    )

    val_data = data_for(
        config,
        "val",
        train=False
    )

    order_generator = torch.Generator()

    validation_generator = (
        torch.Generator()
        .manual_seed(0)
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=order_generator,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda"
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config["batch_size"],
        shuffle=False,
        generator=validation_generator,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda"
    )

    log_path = (
        output_dir
        / "logs"
        / f"{run_id}.csv"
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if (
        log_path.exists()
        and resume is None
    ):
        raise FileExistsError(
            f"Run already exists: {run_id}; "
            "resume or use a different output_dir"
        )

    try:
        git_commit = subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD"
            ],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()

    except (
        FileNotFoundError,
        subprocess.CalledProcessError
    ):
        git_commit = None

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(
            device
        )
    else:
        gpu_name = None

    run_info = {
        "run_id": run_id,
        "model": model_name,
        "train_dataset": "imagenette",
        "train_seed": config["train_seed"],
        "config": config,
        "git_commit": git_commit,
        "parameter_count": parameter_count,
        "device": str(device),
        "GPU": gpu_name,
        "torch_version": torch.__version__,
        "data_identity": identity
    }

    info_path = (
        log_path.with_suffix(
            ".json"
        )
    )

    info_path.write_text(
        json.dumps(
            run_info,
            indent=2
        ),
        encoding="utf-8"
    )

    fields = [
        "epoch",
        "epoch_time",
        "train_loss",
        "val_MAE_hole",
        "val_PSNR_hole",
        "lr",
        "best_epoch",
        "checkpoint_path"
    ]

    if not log_path.exists():
        with log_path.open(
            "w",
            newline=""
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fields
            )

            writer.writeheader()

    elif resume is not None:
        with log_path.open(
            newline=""
        ) as file:
            reader = csv.DictReader(
                file
            )

            previous_rows = [
                row
                for row in reader
                if int(row["epoch"])
                <= start_epoch
            ]

        checkpoint_row = checkpoint.get(
            "log_row"
        )

        has_start_epoch = any(
            int(row["epoch"])
            == start_epoch
            for row in previous_rows
        )

        if (
            checkpoint_row
            and not has_start_epoch
        ):
            previous_rows.append(
                checkpoint_row
            )

        with log_path.open(
            "w",
            newline=""
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fields
            )

            writer.writeheader()

            writer.writerows(
                previous_rows
            )

    for epoch in range(
        start_epoch,
        config["epochs"]
    ):
        if (
            bad_epochs
            >= config[
                "early_stopping_patience"
            ]
        ):
            break

        train_data.epoch = epoch

        epoch_seed = stable_seed(
            config["train_seed"],
            "batch_order",
            epoch
        )

        order_generator.manual_seed(
            epoch_seed
        )

        start_time = time.perf_counter()

        total_loss = 0.0
        total_images = 0

        model.train()

        for batch_index, batch in enumerate(
            train_loader
        ):
            images = batch[
                "image"
            ].to(device)

            masks = batch[
                "mask"
            ].to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            inputs = model_input(
                images,
                masks
            )

            predictions = model(
                inputs
            )

            loss = inpainting_loss(
                predictions,
                images,
                masks
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Non-finite train loss"
                )

            loss.backward()

            gradients_are_valid = all(
                torch.isfinite(
                    parameter.grad
                ).all()
                for parameter
                in model.parameters()
                if parameter.grad
                is not None
            )

            if not gradients_are_valid:
                raise RuntimeError(
                    "Non-finite training gradient"
                )

            optimizer.step()

            batch_size = len(
                images
            )

            total_loss += (
                float(loss.detach())
                * batch_size
            )

            total_images += (
                batch_size
            )

            if (
                batch_index + 1
            ) % 50 == 0:
                average_loss = (
                    total_loss
                    / total_images
                )

                print(
                    f"{model_name}: "
                    f"epoch {epoch + 1}, "
                    f"batch {batch_index + 1}/"
                    f"{len(train_loader)}, "
                    f"loss={average_loss:.5f}",
                    flush=True
                )

        model.eval()

        validation_mae = []
        validation_psnr = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch[
                    "image"
                ].to(device)

                masks = batch[
                    "mask"
                ].to(device)

                inputs = model_input(
                    images,
                    masks
                )

                predictions = model(
                    inputs
                )

                result = tensor_metrics(
                    predictions,
                    images,
                    masks
                )

                validation_mae.extend(
                    result[
                        "MAE_hole"
                    ].cpu().tolist()
                )

                validation_psnr.extend(
                    result[
                        "PSNR_hole"
                    ].cpu().tolist()
                )

        mean_mae = float(
            np.mean(
                validation_mae
            )
        )

        mean_psnr = float(
            np.mean(
                validation_psnr
            )
        )

        scheduler.step(
            mean_mae
        )

        improved = (
            mean_mae
            < best_metric
        )

        if improved:
            best_metric = mean_mae
            best_epoch = epoch + 1
            bad_epochs = 0

        else:
            bad_epochs += 1

        epoch_time = (
            time.perf_counter()
            - start_time
        )

        average_train_loss = (
            total_loss
            / total_images
        )

        current_lr = (
            optimizer.param_groups[
                0
            ]["lr"]
        )

        row = {
            "epoch": epoch + 1,
            "epoch_time": epoch_time,
            "train_loss":
                average_train_loss,
            "val_MAE_hole":
                mean_mae,
            "val_PSNR_hole":
                mean_psnr,
            "lr":
                current_lr,
            "best_epoch":
                best_epoch,
            "checkpoint_path":
                str(best_path)
        }

        payload = {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "epoch":
                epoch + 1,

            "best_metric":
                best_metric,

            "best_epoch":
                best_epoch,

            "bad_epochs":
                bad_epochs,

            "random_states":
                random_states(),

            "config":
                config,

            "run_info":
                run_info,

            "checkpoint_kind":
                "last",

            "data_identity":
                identity,

            "log_row":
                row
        }

        last_path = (
            checkpoint_dir
            / f"{run_id}_last.pt"
        )

        best_snapshot = (
            commit_training_checkpoints(
                last_path,
                best_path,
                payload,
                best_snapshot
            )
        )

        with log_path.open(
            "a",
            newline=""
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fields
            )

            writer.writerow(
                row
            )

        print(
            json.dumps(row),
            flush=True
        )

        if (
            bad_epochs
            >= config[
                "early_stopping_patience"
            ]
        ):
            break

    return best_path
