import argparse
import _bootstrap
from src.utils.config import (
    load_config,
    project_path,
    bank_path,
)

from src.datasets.imagenette import validate_manifests
from src.masks.bank import build_bank, validate_generator
from src.evaluation.plots import mask_figures


def load_configs(config_path):

    main_config = load_config(config_path)

    dataset_config_path = main_config["dataset_config"]
    dataset_config = load_config(dataset_config_path)

    return main_config, dataset_config


def check_dataset(dataset_config):
    
    print("Validating dataset manifests...")

    validate_manifests(dataset_config)

    print("Dataset manifests are valid.")


def check_mask_generator():


    print("Validating mask generator...")

    validate_generator(100)

    print("Mask generator is valid.")


def build_one_mask_bank(
    main_config,
    dataset_config,
    dataset_name,
    split_name,
):
    
    manifest_dir = project_path(
        dataset_config["manifest_dir"]
    )

    manifest_file = (
        manifest_dir
        / f"{dataset_name}_{split_name}.csv"
    )

    output_path = bank_path(
        main_config,
        dataset_name,
        split_name,
    )

    mask_seed = main_config["mask_seed"]

    print(
        f"Building mask bank for "
        f"{dataset_name} / {split_name}..."
    )

    build_bank(
        manifest_file,
        output_path,
        dataset_name,
        split_name,
        mask_seed,
    )


def build_all_mask_banks(
    main_config,
    dataset_config,
):
    """
    Build mask banks for all evaluation splits.
    """

    evaluation_splits = [
        ("imagenette", "val"),
        ("imagenette", "test"),
        ("pets", "test"),
    ]

    for dataset_name, split_name in evaluation_splits:

        build_one_mask_bank(
            main_config=main_config,
            dataset_config=dataset_config,
            dataset_name=dataset_name,
            split_name=split_name,
        )


def create_mask_figures(main_config):
    
    output_dir = main_config["output_dir"]

    mask_bank_dir = main_config.get(
        "mask_bank_dir",
        "data/mask_bank",
    )

    print("Creating mask figures...")

    mask_figures(
        output_dir,
        mask_bank_dir,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the main experiment config",
    )

    args = parser.parse_args()

    main_config, dataset_config = load_configs(
        args.config
    )
    check_dataset(
        dataset_config
    )

    check_mask_generator()
    
    build_all_mask_banks(
        main_config,
        dataset_config,
    )

    create_mask_figures(
        main_config
    )

    print("\nMask bank pipeline completed successfully.")


if __name__ == "__main__":
    main()
