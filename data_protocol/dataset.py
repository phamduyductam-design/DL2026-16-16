"""MVTec AD path discovery without loading test data into training."""

from pathlib import Path


def discover_train_good(dataset_root: Path, category: str) -> list[Path]:
    return sorted((dataset_root / category / "train" / "good").glob("*.png"))


def discover_test(dataset_root: Path, category: str) -> list[Path]:
    return sorted((dataset_root / category / "test").glob("*/*.png"))


def mask_path_for(test_image: Path, category_root: Path) -> Path | None:
    if test_image.parent.name == "good":
        return None
    return category_root / "ground_truth" / test_image.parent.name / f"{test_image.stem}_mask.png"

