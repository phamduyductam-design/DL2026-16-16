"""Configuration loading for the week-1 data protocol.

The checked-in ``.yaml`` file deliberately uses JSON syntax. JSON is valid YAML
1.2, so the configuration remains readable by standard YAML tooling while this
module can parse it without adding a YAML parser as a runtime dependency.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .transforms import TransformConfig


@dataclass(frozen=True)
class DataProtocolConfig:
    categories: tuple[str, ...]
    seed: int
    validation_fraction: float
    data_root_env: str
    transform: TransformConfig
    source: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DataProtocolConfig":
        categories = tuple(str(value) for value in raw.get("categories", ()))
        if not categories:
            raise ValueError("config.categories must contain at least one category")

        validation_fraction = float(raw.get("validation_fraction", 0.2))
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")

        return cls(
            categories=categories,
            seed=int(raw.get("seed", 42)),
            validation_fraction=validation_fraction,
            data_root_env=str(raw.get("data_root_env", "MVTEC_AD_ROOT")),
            transform=TransformConfig.from_dict(raw.get("transform", {})),
            source=dict(raw.get("source", {})),
        )


def load_config(path: str | Path) -> DataProtocolConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected an object in {config_path}")
    return DataProtocolConfig.from_dict(raw)


def resolve_data_root(
    explicit_root: str | Path | None,
    env_name: str = "MVTEC_AD_ROOT",
) -> Path:
    value = explicit_root or os.environ.get(env_name)
    if not value:
        raise ValueError(
            f"Data root is required. Pass --data-root or set the {env_name} environment variable."
        )

    root = Path(value).expanduser().resolve()
    nested = root / "MVTecAD"
    if nested.is_dir():
        root = nested
    return root
