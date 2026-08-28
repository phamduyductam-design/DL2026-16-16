"""Read and write reproducible split manifests."""

import json
from pathlib import Path


def write_manifest(path: Path, train: list[Path], validation: list[Path]) -> None:
    payload = {
        "train": [str(item) for item in train],
        "validation": [str(item) for item in validation],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_manifest(path: Path) -> dict[str, list[Path]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: [Path(item) for item in items] for name, items in payload.items()}

