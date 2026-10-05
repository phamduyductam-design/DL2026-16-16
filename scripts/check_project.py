import _bootstrap
import argparse
import json

import numpy as np
import pandas as pd

from src.utils.config import (
    project_path,
    load_config,
    bank_path,
    file_sha256,
)
from src.utils.checkpoint import (
    load_checkpoint,
    model_state_sha256,
)
from src.datasets.imagenette import validate_manifests
from src.masks.bank import validate_bank
from src.masks.generators import CONDITIONS


VERIFY_ERRORS = (
    OSError,
    ValueError,
    KeyError,
    RuntimeError,
    EOFError,
    AssertionError,
    pd.errors.ParserError,
)


def check(seed=42, config_path="configs/base.yaml", write_status=True):
    config = load_config(config_path)
    config["train_seed"] = seed

    dataset_config = load_config(config["dataset_config"])

    output_dir = project_path(config["output_dir"])
    manifest_dir = project_path(dataset_config["manifest_dir"])

    status = {}
    errors = {}
    banks = {}
    checkpoint_hashes = {}

    def verify(name, operation):
        try:
            result = operation()

            if result is False:
                raise ValueError(
                    "Required artifact missing/inconsistent"
                )

            status[name] = True

        except VERIFY_ERRORS as error:
            status[name] = False
            errors[name] = str(error)

    def verify_manifests():
        return validate_manifests(dataset_config)

    verify(
        "locked_manifests",
        verify_manifests,
    )

    evaluation_splits = [
        ("imagenette", "val"),
        ("imagenette", "test"),
        ("pets", "test"),
    ]

    def verify_bank(dataset, split):
        manifest_file = (
            manifest_dir
            / f"{dataset}_{split}.csv"
        )

        bank_dir = bank_path(
            config,
            dataset,
            split,
        )

        result = validate_bank(
            manifest_file,
            bank_dir,
            dataset,
            split,
            config["mask_seed"],
        )

        banks[f"{dataset}_{split}"] = result

    for dataset, split in evaluation_splits:
        verify(
            f"bank_{dataset}_{split}",
            lambda dataset=dataset, split=split:
                verify_bank(dataset, split),
        )

    identity = None

    if (
        status.get("locked_manifests") is True
        and len(banks) == 3
    ):
        manifest_names = [
            "imagenette_train",
            "imagenette_val",
            "imagenette_test",
            "pets_test",
        ]

        manifest_hashes = {
            name: file_sha256(
                manifest_dir / f"{name}.csv"
            )
            for name in manifest_names
        }

        identity = {
            "manifests": manifest_hashes,
            "banks": banks,
            "audit_sha256": file_sha256(
                manifest_dir / "audit.json"
            ),
        }

    def verify_model_run(model):
        run_id = f"{model}_seed{seed}"

        best_path = (
            output_dir
            / "checkpoints"
            / f"{run_id}_best.pt"
        )

        last_path = (
            output_dir
            / "checkpoints"
            / f"{run_id}_last.pt"
        )

        best = load_checkpoint(best_path)
        last = load_checkpoint(last_path)

        if identity is None:
            raise ValueError(
                "Data identity is not available"
            )

        if (
            best.get("data_identity") != identity
            or last.get("data_identity") != identity
        ):
            raise ValueError(
                "Checkpoint data fingerprint mismatch"
            )

        expected_config = dict(
            config,
            model=model,
        )

        if (
            best.get("config") != expected_config
            or last.get("config") != expected_config
        ):
            raise ValueError(
                "Run configuration mismatch"
            )

        if (
            best.get("checkpoint_kind") != "best"
            or last.get("checkpoint_kind") != "last"
        ):
            raise ValueError(
                "Wrong checkpoint types"
            )

        embedded_best = last["best_checkpoint"]

        best_state_hash = model_state_sha256(
            best["model_state_dict"]
        )

        embedded_state_hash = model_state_sha256(
            embedded_best["model_state_dict"]
        )

        if (
            best["best_epoch"] != last["best_epoch"]
            or best["best_metric"] != last["best_metric"]
            or best_state_hash != best["model_state_sha256"]
            or embedded_state_hash != best["model_state_sha256"]
        ):
            raise ValueError(
                "BEST/LAST are inconsistent"
            )

        if (
            last["epoch"] < config["epochs"]
            and last["bad_epochs"]
            < config["early_stopping_patience"]
        ):
            raise ValueError(
                "Training has not completed"
            )

        log_path = (
            output_dir
            / "logs"
            / f"{run_id}.csv"
        )

        logs = pd.read_csv(log_path)

        if logs.empty:
            raise ValueError(
                "Incomplete training logs"
            )

        if int(logs.epoch.iloc[-1]) != last["epoch"]:
            raise ValueError(
                "Incomplete training logs"
            )

        best_logged_metric = logs.val_MAE_hole.min()

        if abs(
            best_logged_metric - best["best_metric"]
        ) > 1e-10:
            raise ValueError(
                "BEST does not match validation logs"
            )

        sanity_path = (
            output_dir
            / "logs"
            / f"{model}_sanity.json"
        )

        sanity = json.loads(
            sanity_path.read_text()
        )

        if sanity.get("passed") is not True:
            raise ValueError(
                "Missing/failed real sanity check"
            )

        if (
            sanity["final_loss"]
            >= 0.85 * sanity["initial_loss"]
        ):
            raise ValueError(
                "Missing/failed real sanity check"
            )

        checkpoint_hashes[model] = file_sha256(
            best_path
        )

    for model in ["autoencoder", "unet"]:
        verify(
            f"{model}_completed_run",
            lambda model=model:
                verify_model_run(model),
        )

    def verify_results():
        summary_path = (
            output_dir
            / "metrics"
            / "summary.csv"
        )

        per_image_path = (
            output_dir
            / "metrics"
            / "per_image.csv"
        )

        summary = pd.read_csv(summary_path)
        per_image = pd.read_csv(per_image_path)

        expected = {
            (
                model,
                dataset,
                condition,
            )
            for model in [
                "autoencoder",
                "unet",
            ]
            for dataset in [
                "imagenette",
                "pets",
            ]
            for condition in CONDITIONS
        }

        actual = set(
            zip(
                summary.model,
                summary.eval_dataset,
                summary.condition,
            )
        )

        if len(summary) != 48:
            raise ValueError(
                "Expected 48 summary rows"
            )

        if actual != expected:
            raise ValueError(
                "Unexpected summary conditions"
            )

        if len(per_image) != 36000:
            raise ValueError(
                "Expected 36000 per-image rows"
            )

        if not (
            per_image.train_seed == seed
        ).all():
            raise ValueError(
                "Wrong per-image training seed"
            )

        if not (
            summary.train_seed == seed
        ).all():
            raise ValueError(
                "Wrong summary training seed"
            )

        if not (
            per_image.train_dataset
            == "imagenette"
        ).all():
            raise ValueError(
                "Wrong per-image training dataset"
            )

        if not (
            summary.train_dataset
            == "imagenette"
        ).all():
            raise ValueError(
                "Wrong summary training dataset"
            )

        bounded_metrics = [
            "MAE_hole",
            "MSE_hole",
            "actual_ratio",
        ]

        for metric in bounded_metrics:
            values = per_image[metric]

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Invalid measured {metric}"
                )

            if not values.between(0, 1).all():
                raise ValueError(
                    f"Invalid measured {metric}"
                )

        if not np.isfinite(
            per_image.SSIM_full
        ).all():
            raise ValueError(
                "Invalid SSIM-full"
            )

        if not per_image.SSIM_full.between(
            -1,
            1,
        ).all():
            raise ValueError(
                "Invalid SSIM-full"
            )

        if not np.isfinite(
            per_image.inference_ms
        ).all():
            raise ValueError(
                "Invalid inference timing"
            )

        if (
            per_image.inference_ms <= 0
        ).any():
            raise ValueError(
                "Invalid inference timing"
            )

        for metric in [
            "PSNR_hole",
            "PSNR_full",
        ]:
            values = per_image[metric]

            if values.isna().any():
                raise ValueError(
                    f"Invalid measured {metric}"
                )

            if (values < 0).any():
                raise ValueError(
                    f"Invalid measured {metric}"
                )

        if identity is None:
            raise ValueError(
                "Cannot certify results without valid complete runs"
            )

        if len(checkpoint_hashes) != 2:
            raise ValueError(
                "Cannot certify results without valid complete runs"
            )

        grouped = per_image.groupby(
            [
                "model",
                "eval_dataset",
            ]
        )

        for (model, dataset), rows in grouped:
            manifest_path = (
                manifest_dir
                / f"{dataset}_test.csv"
            )

            manifest = pd.read_csv(
                manifest_path
            )

            expected_pairs = {
                (
                    image_id,
                    condition,
                )
                for image_id
                in manifest.image_id
                for condition
                in CONDITIONS
            }

            actual_pairs = set(
                zip(
                    rows.image_id,
                    rows.condition,
                )
            )

            if len(rows) != len(
                expected_pairs
            ):
                raise ValueError(
                    "Wrong per-image evaluation pairs"
                )

            if actual_pairs != expected_pairs:
                raise ValueError(
                    "Wrong per-image evaluation pairs"
                )

            if not (
                rows.checkpoint_sha256
                == checkpoint_hashes[model]
            ).all():
                raise ValueError(
                    "Stale evaluation checkpoint"
                )

            bank_info = banks[
                f"{dataset}_test"
            ]

            if not (
                rows.bank_sha256
                == bank_info["bank_sha256"]
            ).all():
                raise ValueError(
                    "Stale evaluation bank"
                )

            manifest_hash = identity[
                "manifests"
            ][f"{dataset}_test"]

            if not (
                rows.manifest_sha256
                == manifest_hash
            ).all():
                raise ValueError(
                    "Stale evaluation manifest"
                )

            metadata_path = (
                bank_path(
                    config,
                    dataset,
                    "test",
                )
                / "metadata.csv"
            )

            bank_metadata = pd.read_csv(
                metadata_path
            )

            keys = [
                "image_id",
                "mask_id",
                "condition",
                "pattern",
                "size",
                "actual_ratio",
            ]

            observed = (
                rows[keys]
                .sort_values(
                    [
                        "image_id",
                        "condition",
                    ]
                )
                .reset_index(drop=True)
            )

            locked = (
                bank_metadata[keys]
                .sort_values(
                    [
                        "image_id",
                        "condition",
                    ]
                )
                .reset_index(drop=True)
            )

            pd.testing.assert_frame_equal(
                observed,
                locked,
                check_dtype=False,
                rtol=1e-9,
                atol=1e-10,
            )

        from src.evaluation.evaluate import aggregate

        calculated = (
            aggregate(per_image)
            .sort_values(
                [
                    "model",
                    "eval_dataset",
                    "condition",
                ]
            )
            .reset_index(drop=True)
        )

        measured = (
            summary
            .sort_values(
                [
                    "model",
                    "eval_dataset",
                    "condition",
                ]
            )
            .reset_index(drop=True)
        )

        pd.testing.assert_frame_equal(
            calculated,
            measured,
            check_dtype=False,
            rtol=1e-8,
            atol=1e-10,
        )

    verify(
        "verified_results",
        verify_results,
    )

    def verify_failure_cases():
        selections_path = (
            output_dir
            / "metrics"
            / "qualitative_selections.csv"
        )

        if not selections_path.is_file():
            return False

        selections = pd.read_csv(
            selections_path
        )

        if len(selections) != 72:
            return False

        for dataset in [
            "imagenette",
            "pets",
        ]:
            for condition in CONDITIONS:
                figure_path = (
                    output_dir
                    / "figures"
                    / f"{dataset}_{condition}_failure_grid.png"
                )

                if not figure_path.is_file():
                    return False

        return True

    verify(
        "failure_cases",
        verify_failure_cases,
    )

    def verify_report():
        report_path = project_path(
            "report/results.md"
        )

        return report_path.is_file()

    verify(
        "measured_report",
        verify_report,
    )

    result = {
        "checks": status,
        "errors": errors,
    }

    if write_status:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        status_path = (
            output_dir
            / "project_status.json"
        )

        status_path.write_text(
            json.dumps(
                result,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    return status


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
    )

    args = parser.parse_args()

    status = check(
        seed=args.seed,
        config_path=args.config,
    )

    success = all(
        status.values()
    )

    raise SystemExit(
        0 if success else 1
    )


if __name__ == "__main__":
    main()
