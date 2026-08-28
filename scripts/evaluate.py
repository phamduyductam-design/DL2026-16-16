"""Unified evaluation entry point owned by Người 5."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anomaly scores and maps")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    print(f"Evaluation entry point: predictions={args.predictions}, config={args.config}")


if __name__ == "__main__":
    main()

