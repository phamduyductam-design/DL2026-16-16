import argparse
import subprocess
import sys
import _bootstrap
from src.utils.config import project_path, load_config
def run_command(command):
    subprocess.run(
        [sys.executable] + command,
        cwd=project_path("."),
        check=True
    )

def prepare_dataset(dataset_config_path):
    
    dataset_config = load_config(dataset_config_path)

    manifest_dir = project_path(dataset_config["manifest_dir"])

    manifest_names = [
        "imagenette_train",
        "imagenette_val",
        "imagenette_test",
        "pets_test",
    ]

    manifest_exists = [
        (manifest_dir / f"{name}.csv").exists()
        for name in manifest_names
    ]

    if not any(manifest_exists):
        print("Preparing dataset manifests...")

        run_command([
            "scripts/prepare_data.py",
            "--config",
            dataset_config_path
        ])

    else:
        print("Dataset manifests already exist. Skipping data preparation.")


def run_tests():
    print("Running tests...")

    run_command([
        "-m",
        "pytest",
        "-q"
    ])


def build_mask_bank(config_path):
    print("Building mask bank...")

    run_command([
        "scripts/build_mask_bank.py",
        "--config",
        config_path
    ])


def train_model(model_name, config_path, seed, output_dir):
    

    checkpoint_dir = output_dir / "checkpoints"

    last_checkpoint = (
        checkpoint_dir
        / f"{model_name}_seed{seed}_last.pt"
    )

    command = [
        "scripts/train.py",
        "--model",
        model_name,
        "--config",
        config_path,
        "--seed",
        str(seed),
    ]

    if last_checkpoint.exists():
        print(f"Resuming {model_name} from {last_checkpoint}")

        command += [
            "--resume",
            str(last_checkpoint)
        ]

    else:
        print(f"Training {model_name} from scratch...")

    run_command(command)


def evaluate_model(
    model_name,
    dataset_name,
    seed,
    batch_size,
    dataset_config_path,
    output_dir
):


    best_checkpoint = (
        output_dir
        / "checkpoints"
        / f"{model_name}_seed{seed}_best.pt"
    )

    print(
        f"Evaluating {model_name} on {dataset_name}..."
    )

    run_command([
        "scripts/evaluate.py",

        "--checkpoint",
        str(best_checkpoint),

        "--dataset",
        dataset_name,

        "--dataset-config",
        dataset_config_path,

        "--batch-size",
        str(batch_size),
    ])


def combine_results(seed, output_dir):
    print("Combining evaluation results...")

    run_command([
        "scripts/evaluate.py",
        "--combine",
        "--seed",
        str(seed),
        "--output",
        str(output_dir),
    ])


def plot_results(config, output_dir):
    print("Plotting results...")

    mask_bank_dir = config.get(
        "mask_bank_dir",
        "data/mask_bank"
    )

    run_command([
        "scripts/plot_results.py",

        "--output",
        str(output_dir),

        "--dataset-config",
        config["dataset_config"],

        "--mask-bank-dir",
        mask_bank_dir,
    ])


def final_check(seed, config_path):
    print("Running final project check...")

    run_command([
        "scripts/check_project.py",
        "--seed",
        str(seed),
        "--config",
        config_path,
    ])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--config",
        default="configs/base.yaml"
    )

    args = parser.parse_args()

    # Load main config
    config = load_config(args.config)

    dataset_config_path = config["dataset_config"]

    output_dir = project_path(
        config["output_dir"]
    )

    

    prepare_dataset(
        dataset_config_path
    )

    
    run_tests()



    build_mask_bank(
        args.config
    )
    models = [
        "autoencoder",
        "unet"
    ]

    datasets = [
        "imagenette",
        "pets"
    ]

    for model_name in models:

        train_model(
            model_name=model_name,
            config_path=args.config,
            seed=args.seed,
            output_dir=output_dir,
        )

        for dataset_name in datasets:

            evaluate_model(
                model_name=model_name,
                dataset_name=dataset_name,
                seed=args.seed,
                batch_size=config["batch_size"],
                dataset_config_path=dataset_config_path,
                output_dir=output_dir,
            )


    combine_results(
        seed=args.seed,
        output_dir=output_dir,
    )

    plot_results(
        config=config,
        output_dir=output_dir,
    )


    final_check(
        seed=args.seed,
        config_path=args.config
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
