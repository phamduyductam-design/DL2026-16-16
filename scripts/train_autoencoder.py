"""Training entry point owned by Người 2."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Autoencoder on MVTec normal images")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--category", required=True)
    args = parser.parse_args()
    print(f"Autoencoder training entry point: category={args.category}, config={args.config}")


if __name__ == "__main__":
    main()

