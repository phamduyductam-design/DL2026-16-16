"""Transform configuration shared by all model owners."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformConfig:
    image_size: int = 256
    normalization: str = "imagenet"
    augmentation: str = "light"

    def validate(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.normalization != "imagenet":
            raise ValueError("all pretrained backbones must use ImageNet normalization")
        if self.augmentation not in {"none", "light"}:
            raise ValueError("augmentation must be none or light")

