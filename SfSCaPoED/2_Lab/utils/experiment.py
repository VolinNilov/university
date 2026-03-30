from __future__ import annotations

import numpy as np

# Квантили t(0.975, df) = scipy.stats.t.ppf(0.975, df) — если scipy нет, для ЛР2 хватает df=4 (N=5) и df=24 (N=25)
_T_PPF_975_FALLBACK = {
    4: 2.7764451051977987,
    24: 2.0638985616282405,
}


def _t_critical_two_sided(df: int, alpha: float) -> float:
    p = 1.0 - alpha / 2.0
    try:
        from scipy import stats

        return float(stats.t.ppf(p, df=df))
    except ImportError:
        if not np.isclose(alpha, 0.05) or abs(p - 0.975) > 1e-9:
            raise ImportError(
                "Нужен пакет scipy. В активном окружении: pip install scipy"
            ) from None
        if df in _T_PPF_975_FALLBACK:
            return _T_PPF_975_FALLBACK[df]
        raise ImportError(
            f"Нужен пакет scipy (pip install scipy). Нет встроенной таблицы для df={df}."
        ) from None


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


def ensemble_experimental(
    x_th: np.ndarray,
    y_th: np.ndarray,
    n_runs: int,
    sigma: float,
    seed_base: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_all = np.zeros((n_runs, len(x_th)))
    y_all = np.zeros((n_runs, len(y_th)))
    for k in range(n_runs):
        rng = np.random.default_rng(seed_base + k * 10007)
        xe, ye = experimental_trajectory(x_th, y_th, sigma, rng)
        x_all[k, :] = xe
        y_all[k, :] = ye
    return x_all, y_all


def mean_and_ci_student_t(
    data: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, _ = data.shape
    if n < 2:
        raise ValueError("Для интервала Стьюдента нужно N >= 2")
    mean = np.mean(data, axis=0)
    s = np.std(data, axis=0, ddof=1)
    t_crit = _t_critical_two_sided(n - 1, alpha)
    delta = t_crit * s / np.sqrt(n)
    return mean, mean - delta, mean + delta
