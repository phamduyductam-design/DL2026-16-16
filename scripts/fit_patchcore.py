"""PatchCore fitting entry point owned by Người 4."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PatchCore memory bank")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--category", required=True)
    args = parser.parse_args()
    print(f"PatchCore fitting entry point: category={args.category}, config={args.config}")


if __name__ == "__main__":
    main()

