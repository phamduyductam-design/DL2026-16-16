"""Deterministic normal-only train/validation splitting."""

import random
from collections.abc import Sequence
from pathlib import Path


def deterministic_split(
    paths: Sequence[Path], validation_ratio: float = 0.20, seed: int = 42
) -> tuple[list[Path], list[Path]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    ordered = sorted(paths)
    shuffled = ordered.copy()
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, round(len(shuffled) * validation_ratio))
    validation = sorted(shuffled[:validation_size])
    train = sorted(shuffled[validation_size:])
    return train, validation


def assert_disjoint(*splits: Sequence[Path]) -> None:
    seen: set[Path] = set()
    for split in splits:
        current = set(split)
        if seen.intersection(current):
            raise ValueError("data leakage detected: splits overlap")
        seen.update(current)

