"""Simple gradient-energy baseline used as a lower-bound comparison."""

import numpy as np


def gradient_anomaly_map(image: np.ndarray) -> np.ndarray:
    gray = image.astype(np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    grad_y, grad_x = np.gradient(gray)
    energy = np.hypot(grad_x, grad_y)
    scale = float(energy.max())
    return energy / scale if scale > 0 else np.zeros_like(energy)


def image_score(anomaly_map: np.ndarray, top_fraction: float = 0.01) -> float:
    flat = np.asarray(anomaly_map, dtype=np.float32).ravel()
    count = max(1, round(flat.size * top_fraction))
    return float(np.partition(flat, -count)[-count:].mean())

