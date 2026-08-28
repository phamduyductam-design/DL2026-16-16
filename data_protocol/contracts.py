"""Canonical interfaces shared by data, models, evaluation, and deployment."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Sample:
    image: np.ndarray
    label: int
    mask: np.ndarray
    category: str
    defect_type: str
    image_path: Path

    def validate(self) -> None:
        if self.label not in {0, 1}:
            raise ValueError("label must be 0 (good) or 1 (defect)")
        if self.mask.ndim != 2:
            raise ValueError("mask must be a two-dimensional array")
        if self.label == 0 and np.any(self.mask):
            raise ValueError("good samples must use an all-zero mask")


@dataclass(frozen=True)
class Prediction:
    image_score: float
    anomaly_map: np.ndarray
    model_name: str
    category: str
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    def validate(self) -> None:
        if not np.isfinite(self.image_score):
            raise ValueError("image_score must be finite")
        if self.anomaly_map.ndim != 2:
            raise ValueError("anomaly_map must be a two-dimensional array")

