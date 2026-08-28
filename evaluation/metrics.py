"""Threshold-dependent pixel localization metrics."""

import numpy as np


def _binary_arrays(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target shapes must match")
    return pred, truth


def dice_score(prediction: np.ndarray, target: np.ndarray, epsilon: float = 1e-8) -> float:
    pred, truth = _binary_arrays(prediction, target)
    intersection = np.logical_and(pred, truth).sum()
    return float((2 * intersection + epsilon) / (pred.sum() + truth.sum() + epsilon))


def intersection_over_union(
    prediction: np.ndarray, target: np.ndarray, epsilon: float = 1e-8
) -> float:
    pred, truth = _binary_arrays(prediction, target)
    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    return float((intersection + epsilon) / (union + epsilon))

