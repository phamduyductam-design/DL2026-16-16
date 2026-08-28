"""Threshold calibration using validation-normal scores only."""

import numpy as np


def percentile_threshold(normal_scores: np.ndarray, percentile: float = 99.0) -> float:
    scores = np.asarray(normal_scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("normal_scores cannot be empty")
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be between 0 and 100")
    return float(np.percentile(scores, percentile))

