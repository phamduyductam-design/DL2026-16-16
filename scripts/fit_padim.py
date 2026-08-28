"""PaDiM fitting entry point owned by Người 3."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit PaDiM statistics on normal features")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--category", required=True)
    args = parser.parse_args()
    print(f"PaDiM fitting entry point: category={args.category}, config={args.config}")


if __name__ == "__main__":
    main()

