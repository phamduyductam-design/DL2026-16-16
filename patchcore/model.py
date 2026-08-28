"""Nearest-neighbor scoring for PatchCore patch embeddings."""

import numpy as np


def nearest_neighbor_scores(features: np.ndarray, memory_bank: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or memory_bank.ndim != 2:
        raise ValueError("features and memory_bank must be two-dimensional")
    if features.shape[1] != memory_bank.shape[1]:
        raise ValueError("feature dimensions must match")
    distances = np.linalg.norm(features[:, None, :] - memory_bank[None, :, :], axis=2)
    return distances.min(axis=1)

