"""Helpers for labeling representative model failure cases."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ErrorCase:
    image_path: Path
    model_name: str
    category: str
    kind: Literal["false_positive", "false_negative", "localization_error"]
    note: str

