"""Deterministic image preprocessing shared by all model workstreams."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class TransformConfig:
    height: int = 256
    width: int = 256
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    horizontal_flip_probability: float = 0.0
    seed: int = 42

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TransformConfig":
        resolution = raw.get("resolution", [256, 256])
        if not isinstance(resolution, list) or len(resolution) != 2:
            raise ValueError("transform.resolution must be [height, width]")

        mean = tuple(float(value) for value in raw.get("mean", cls.mean))
        std = tuple(float(value) for value in raw.get("std", cls.std))
        if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
            raise ValueError("transform mean/std must contain three channels and std must be positive")

        probability = float(raw.get("horizontal_flip_probability", 0.0))
        if not 0.0 <= probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be between 0 and 1")

        return cls(
            height=int(resolution[0]),
            width=int(resolution[1]),
            mean=mean,
            std=std,
            horizontal_flip_probability=probability,
            seed=int(raw.get("seed", 42)),
        )


class MVTecTransform:
    """Return normalized CHW float32 arrays and nearest-neighbor binary masks."""

    def __init__(self, config: TransformConfig) -> None:
        self.config = config
        self._mean = np.asarray(config.mean, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.asarray(config.std, dtype=np.float32).reshape(3, 1, 1)

    def _should_flip(self, sample_id: str) -> bool:
        token = f"{self.config.seed}:{sample_id}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / float(2**64)
        return value < self.config.horizontal_flip_probability

    def image(self, image: Image.Image, sample_id: str, training: bool) -> np.ndarray:
        image = image.convert("RGB")
        if training and self._should_flip(sample_id):
            image = ImageOps.mirror(image)
        image = image.resize((self.config.width, self.config.height), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        array = np.transpose(array, (2, 0, 1))
        return ((array - self._mean) / self._std).astype(np.float32, copy=False)

    def mask(self, mask: Image.Image) -> np.ndarray:
        mask = mask.convert("L").resize(
            (self.config.width, self.config.height),
            Image.Resampling.NEAREST,
        )
        return (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8, copy=False)[None, ...]
