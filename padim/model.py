"""NumPy reference implementation of PaDiM distribution fitting and scoring."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PadimStatistics:
    mean: np.ndarray
    precision: np.ndarray


def fit_statistics(features: np.ndarray, regularization: float = 1e-2) -> PadimStatistics:
    if features.ndim != 4:
        raise ValueError("features must have shape [samples, height, width, dimensions]")
    mean = features.mean(axis=0)
    centered = features - mean
    covariance = np.einsum("nhwd,nhwe->hwde", centered, centered)
    covariance /= max(1, features.shape[0] - 1)
    identity = np.eye(features.shape[-1], dtype=features.dtype)
    precision = np.linalg.pinv(covariance + regularization * identity)
    return PadimStatistics(mean=mean, precision=precision)


def score_features(features: np.ndarray, statistics: PadimStatistics) -> np.ndarray:
    difference = features - statistics.mean
    squared = np.einsum("hwd,hwde,hwe->hw", difference, statistics.precision, difference)
    return np.sqrt(np.maximum(squared, 0.0))

