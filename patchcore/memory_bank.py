"""Memory-bank construction with deterministic subset selection."""

import numpy as np


def build_memory_bank(features: np.ndarray, coreset_ratio: float, seed: int = 42) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("features must have shape [patches, dimensions]")
    if not 0.0 < coreset_ratio <= 1.0:
        raise ValueError("coreset_ratio must be in (0, 1]")
    size = max(1, round(len(features) * coreset_ratio))
    indices = np.random.default_rng(seed).choice(len(features), size=size, replace=False)
    return features[np.sort(indices)].copy()

