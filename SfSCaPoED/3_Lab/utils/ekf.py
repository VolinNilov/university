from __future__ import annotations

import numpy as np


def rk4_one_step(eom, t: float, x: np.ndarray, dt: float) -> np.ndarray:
    k1 = dt * eom(t, x)
    k2 = dt * eom(t + dt / 2, x + k1 / 2)
    k3 = dt * eom(t + dt / 2, x + k2 / 2)
    k4 = dt * eom(t + dt, x + k3)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def jacobian_phi(
    eom, t: float, x: np.ndarray, dt: float, eps: float = 1e-5
) -> np.ndarray:
    n = x.shape[0]
    phi0 = rk4_one_step(eom, t, x, dt)
    F = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        phi_p = rk4_one_step(eom, t, x + dx, dt)
        F[:, i] = (phi_p - phi0) / eps
    return F


def run_ekf(
    z_x: np.ndarray,
    z_y: np.ndarray,
    t: np.ndarray,
    dt: float,
    motion,
    x0: np.ndarray,
    P0: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    eom = motion.equations_of_motion
    n = len(t)
    if len(z_x) != n or len(z_y) != n:
        raise ValueError("Длины z_x, z_y и t должны совпадать")

    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    I4 = np.eye(4)

    xs = np.zeros((n, 4))
    x = np.asarray(x0, dtype=float).copy().reshape(4)
    P = np.asarray(P0, dtype=float).copy()

    z = np.array([z_x[0], z_y[0]])
    S = H @ P @ H.T + R
    K = np.linalg.solve(S, (P @ H.T).T).T
    x = x + K @ (z - H @ x)
    P = (I4 - K @ H) @ P
    xs[0] = x

    for k in range(1, n):
        tk = float(t[k - 1])
        F = jacobian_phi(eom, tk, x, dt)
        x_pred = rk4_one_step(eom, tk, x, dt)
        P_pred = F @ P @ F.T + Q
        z = np.array([z_x[k], z_y[k]])
        S = H @ P_pred @ H.T + R
        K = np.linalg.solve(S, (P_pred @ H.T).T).T
        x = x_pred + K @ (z - H @ x_pred)
        P = (I4 - K @ H) @ P_pred
        xs[k] = x

    return xs
