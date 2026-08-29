"""Leakage-safe catalog, split, manifest and dataset access for MVTec AD."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from PIL import Image

from .transforms import MVTecTransform

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ProtocolError(RuntimeError):
    """Raised when local data violates the frozen project protocol."""


@dataclass(frozen=True)
class SampleRecord:
    category: str
    sample_id: str
    image_path: Path
    official_split: str
    protocol_split: str
    defect_type: str
    is_anomaly: bool
    mask_path: Path | None = None

    def manifest_entry(self, data_root: Path, include_label: bool) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "category": self.category,
            "sample_id": self.sample_id,
            "image_path": self.image_path.relative_to(data_root).as_posix(),
            "official_split": self.official_split,
            "protocol_split": self.protocol_split,
        }
        if include_label:
            entry["defect_type"] = self.defect_type
            entry["is_anomaly"] = self.is_anomaly
            entry["mask_path"] = (
                self.mask_path.relative_to(data_root).as_posix() if self.mask_path else None
            )
        return entry


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class MVTECADCatalog:
    """Discover samples without exposing official test labels to model inference."""

    def __init__(self, data_root: str | Path, categories: Sequence[str]) -> None:
        root = Path(data_root).expanduser().resolve()
        if (root / "MVTecAD").is_dir():
            root = (root / "MVTecAD").resolve()
        self.data_root = root
        self.categories = tuple(categories)
        if not self.categories:
            raise ValueError("At least one category is required")

    def scan(self) -> tuple[list[SampleRecord], list[SampleRecord]]:
        official_train: list[SampleRecord] = []
        official_test: list[SampleRecord] = []
        errors: list[str] = []

        for category in self.categories:
            category_root = self.data_root / category
            if not category_root.is_dir():
                errors.append(f"Missing category directory: {category_root}")
                continue

            train_root = category_root / "train"
            unexpected_train_dirs = sorted(
                path.name for path in train_root.iterdir() if path.is_dir() and path.name != "good"
            ) if train_root.is_dir() else []
            if unexpected_train_dirs:
                errors.append(
                    f"{category}/train contains non-good directories: {unexpected_train_dirs}"
                )

            train_images = _image_files(train_root / "good")
            if not train_images:
                errors.append(f"No normal training images found for {category}")
            for image_path in train_images:
                official_train.append(
                    SampleRecord(
                        category=category,
                        sample_id=f"{category}/train/good/{image_path.stem}",
                        image_path=image_path,
                        official_split="train",
                        protocol_split="unassigned",
                        defect_type="good",
                        is_anomaly=False,
                    )
                )

            test_root = category_root / "test"
            if not test_root.is_dir():
                errors.append(f"Missing test directory: {test_root}")
                continue

            defect_dirs = sorted(path for path in test_root.iterdir() if path.is_dir())
            if not defect_dirs:
                errors.append(f"No test defect directories found for {category}")
            for defect_dir in defect_dirs:
                defect_type = defect_dir.name
                for image_path in _image_files(defect_dir):
                    is_anomaly = defect_type != "good"
                    mask_path: Path | None = None
                    if is_anomaly:
                        candidate = (
                            category_root
                            / "ground_truth"
                            / defect_type
                            / f"{image_path.stem}_mask.png"
                        ).resolve()
                        if not candidate.is_file():
                            errors.append(
                                f"Missing mask for {image_path.relative_to(self.data_root)}: "
                                f"{candidate.relative_to(self.data_root)}"
                            )
                        else:
                            mask_path = candidate

                    official_test.append(
                        SampleRecord(
                            category=category,
                            sample_id=f"{category}/test/{defect_type}/{image_path.stem}",
                            image_path=image_path,
                            official_split="test",
                            protocol_split="test",
                            defect_type=defect_type,
                            is_anomaly=is_anomaly,
                            mask_path=mask_path,
                        )
                    )

        if errors:
            preview = "\n - ".join(errors[:30])
            suffix = f"\n - ... and {len(errors) - 30} more" if len(errors) > 30 else ""
            raise ProtocolError(f"Dataset structure validation failed:\n - {preview}{suffix}")

        return official_train, official_test


def deterministic_normal_split(
    official_train: Sequence[SampleRecord],
    validation_fraction: float,
    seed: int,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    by_category: dict[str, list[SampleRecord]] = {}
    for record in official_train:
        if record.official_split != "train" or record.is_anomaly or record.mask_path is not None:
            raise ProtocolError(f"Invalid sample in normal training pool: {record.sample_id}")
        by_category.setdefault(record.category, []).append(record)

    train: list[SampleRecord] = []
    validation: list[SampleRecord] = []
    for category in sorted(by_category):
        records = by_category[category]
        if len(records) < 2:
            raise ProtocolError(f"Category {category} needs at least two normal train images")

        ranked = sorted(
            records,
            key=lambda record: hashlib.sha256(
                f"{seed}:{record.sample_id}".encode("utf-8")
            ).digest(),
        )
        validation_count = max(1, int(round(len(ranked) * validation_fraction)))
        validation_count = min(validation_count, len(ranked) - 1)
        validation_ids = {record.sample_id for record in ranked[:validation_count]}

        for record in sorted(records, key=lambda item: item.sample_id):
            if record.sample_id in validation_ids:
                validation.append(replace(record, protocol_split="validation"))
            else:
                train.append(replace(record, protocol_split="train"))

    return train, validation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_protocol_integrity(
    train: Sequence[SampleRecord],
    validation: Sequence[SampleRecord],
    official_test: Sequence[SampleRecord],
    *,
    hash_audit: bool = False,
) -> None:
    named_splits = {"train": train, "validation": validation, "test": official_test}
    path_sets = {
        name: {record.image_path.resolve() for record in records}
        for name, records in named_splits.items()
    }
    if path_sets["train"] & path_sets["validation"]:
        raise ProtocolError("Train and validation contain the same image path")
    if (path_sets["train"] | path_sets["validation"]) & path_sets["test"]:
        raise ProtocolError("Official test image leaked into train/validation")

    all_ids: list[str] = []
    for record in list(train) + list(validation):
        if record.official_split != "train" or record.is_anomaly or record.mask_path is not None:
            raise ProtocolError(f"Non-normal or test metadata leaked into {record.protocol_split}")
        all_ids.append(record.sample_id)
    all_ids.extend(record.sample_id for record in official_test)
    if len(all_ids) != len(set(all_ids)):
        raise ProtocolError("Duplicate sample_id detected")

    for record in official_test:
        if record.official_split != "test" or record.protocol_split != "test":
            raise ProtocolError(f"Invalid official test record: {record.sample_id}")
        if record.is_anomaly and (record.mask_path is None or not record.mask_path.is_file()):
            raise ProtocolError(f"Missing anomaly mask: {record.sample_id}")
        if not record.is_anomaly and record.mask_path is not None:
            raise ProtocolError(f"Good test image unexpectedly has a mask: {record.sample_id}")

    if hash_audit:
        seen: dict[str, tuple[str, str]] = {}
        for split_name, records in named_splits.items():
            for record in records:
                checksum = _sha256(record.image_path)
                previous = seen.get(checksum)
                if previous and previous[0] != split_name:
                    raise ProtocolError(
                        "Byte-identical image occurs across protocol splits: "
                        f"{previous[1]} ({previous[0]}) and {record.sample_id} ({split_name})"
                    )
                seen[checksum] = (split_name, record.sample_id)


def verify_image_files(records: Iterable[SampleRecord], include_masks: bool = True) -> None:
    for record in records:
        try:
            with Image.open(record.image_path) as image:
                image.verify()
            if include_masks and record.mask_path:
                with Image.open(record.mask_path) as mask:
                    mask.verify()
        except Exception as exc:  # Pillow exposes several decoder-specific errors.
            raise ProtocolError(f"Corrupt or unreadable image for {record.sample_id}: {exc}") from exc


def file_checksum_inventory(
    data_root: str | Path,
    categories: Sequence[str],
) -> dict[str, Any]:
    """Build a deterministic SHA-256 inventory for every selected category file."""

    root = Path(data_root).resolve()
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for category in sorted(categories):
        category_root = root / category
        if not category_root.is_dir():
            raise ProtocolError(f"Cannot checksum missing category: {category_root}")
        for path in sorted(item for item in category_root.rglob("*") if item.is_file()):
            size = path.stat().st_size
            total_bytes += size
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": size,
                    "sha256": _sha256(path),
                }
            )
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "categories": sorted(categories),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def build_manifest(
    data_root: str | Path,
    train: Sequence[SampleRecord],
    validation: Sequence[SampleRecord],
    official_test: Sequence[SampleRecord],
    *,
    seed: int,
    validation_fraction: float,
    source: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(data_root).resolve()
    categories = sorted({record.category for record in list(train) + list(validation)})
    return {
        "schema_version": 1,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "categories": categories,
        "source": source or {},
        "transform": transform or {},
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(official_test),
        },
        "splits": {
            "train": [record.manifest_entry(root, include_label=False) for record in train],
            "validation": [
                record.manifest_entry(root, include_label=False) for record in validation
            ],
            "test": [record.manifest_entry(root, include_label=False) for record in official_test],
        },
    }


def build_evaluation_manifest(
    data_root: str | Path,
    official_test: Sequence[SampleRecord],
    *,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep test labels and mask paths in an explicitly evaluation-only artifact."""

    root = Path(data_root).resolve()
    return {
        "schema_version": 1,
        "usage": "evaluation_only",
        "source": source or {},
        "counts": {
            "test": len(official_test),
            "test_anomaly": sum(record.is_anomaly for record in official_test),
            "test_good": sum(not record.is_anomaly for record in official_test),
        },
        "test_annotations": [
            record.manifest_entry(root, include_label=True) for record in official_test
        ],
    }


def write_manifest(manifest: dict[str, Any], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)
    return path.resolve()


class MVTECDataset:
    """Load RGB images while keeping official test annotations evaluation-only."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        *,
        transform: MVTecTransform | None = None,
        training: bool = False,
        evaluation: bool = False,
        image_loader: Callable[[Path], Image.Image] | None = None,
    ) -> None:
        if training and evaluation:
            raise ValueError("A dataset cannot be both training and evaluation")
        if training and any(record.protocol_split != "train" for record in records):
            raise ProtocolError("training=True accepts only protocol train records")
        if evaluation and any(record.official_split != "test" for record in records):
            raise ProtocolError("evaluation=True accepts only official test records")

        self.records = list(records)
        self.transform = transform
        self.training = training
        self.evaluation = evaluation
        self.image_loader = image_loader or self._load_rgb

    @staticmethod
    def _load_rgb(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = self.image_loader(record.image_path)
        if self.transform:
            image_value: Any = self.transform.image(image, record.sample_id, self.training)
        else:
            image_value = image

        sample: dict[str, Any] = {
            "image": image_value,
            "category": record.category,
            "sample_id": record.sample_id,
            "source_split": record.protocol_split,
            "image_path": str(record.image_path),
        }

        if self.evaluation:
            sample["defect_type"] = record.defect_type
            sample["is_anomaly"] = record.is_anomaly
            sample["mask_path"] = str(record.mask_path) if record.mask_path else None
            if record.mask_path:
                with Image.open(record.mask_path) as mask:
                    mask = mask.convert("L")
                sample["mask"] = self.transform.mask(mask) if self.transform else mask
            else:
                sample["mask"] = None

        return sample
