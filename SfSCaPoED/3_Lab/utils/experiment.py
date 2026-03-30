from __future__ import annotations

import numpy as np


def cumulative_noise(n_steps: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    zeta = rng.normal(0.0, sigma, size=n_steps)
    return np.cumsum(zeta)


def experimental_trajectory(
    x_th: np.ndarray,
    y_th: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(x_th)
    return x_th + cumulative_noise(n, sigma, rng), y_th + cumulative_noise(n, sigma, rng)


def one_experimental_trajectory(
    x_th: np.ndarray, y_th: np.ndarray, sigma: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return experimental_trajectory(x_th, y_th, sigma, rng)
