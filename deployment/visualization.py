"""Small dependency-free visualization helpers."""

import numpy as np


def colorize_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
    heat = np.clip(np.asarray(anomaly_map, dtype=np.float32), 0.0, 1.0)
    red = heat
    green = 1.0 - np.abs(2.0 * heat - 1.0)
    blue = 1.0 - heat
    return np.stack([red, green, blue], axis=-1)


def overlay_heatmap(image: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.asarray(image, dtype=np.float32)
    if base.max() > 1.0:
        base = base / 255.0
    heatmap = colorize_heatmap(anomaly_map)
    if base.shape != heatmap.shape:
        raise ValueError("image and colorized anomaly map shapes must match")
    return np.clip((1.0 - alpha) * base + alpha * heatmap, 0.0, 1.0)

