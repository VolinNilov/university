import numpy as np

from PyQt6.QtCore import QObject, pyqtSignal

from utils.solvers import RungeKutta4Method
from utils.model import ConstraintMotionModel
from utils.experiment import one_experimental_trajectory
from utils.ekf import run_ekf

_F_TOL = 1e-3
_V_TOL = 1e-3


class AppModel(QObject):

    calculationFinished = pyqtSignal(dict)
    calculationStarted = pyqtSignal()
    calculationError = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.params = {}

    def set_parameters(self, params):
        self.params = params

    def run_calculations(self):
        try:
            self.calculationStarted.emit()

            model = ConstraintMotionModel(
                gravity=self.params["gravity"],
                friction_coeff=self.params["friction_coeff"],
                mass=self.params["mass"],
            )

            x0_ic = self.params["x0"]
            y0_ic = self.params["y0"]
            vx0 = self.params["vx0"]
            vy0 = self.params["vy0"]

            f0 = ConstraintMotionModel.f(x0_ic, y0_ic)
            if abs(f0) > _F_TOL:
                raise ValueError(
                    f"Точка ({x0_ic}, {y0_ic}) не на связи: f ≈ {f0:.6g} (допуск |f| ≤ {_F_TOL})."
                )
            gfx, gfy = ConstraintMotionModel.grad_f(x0_ic, y0_ic)
            fdot = gfx * vx0 + gfy * vy0
            if abs(fdot) > _V_TOL:
                raise ValueError(
                    f"Скорость не касательна к связи: ∇f·v ≈ {fdot:.6g} (допуск ≤ {_V_TOL})."
                )

            initial_conditions = [x0_ic, y0_ic, vx0, vy0]

            t_start = self.params["t_start"]
            t_end = self.params["t_end"]
            dt = float(self.params["dt"])
            if t_end <= t_start:
                raise ValueError(
                    f"Конец интервала (t_end={t_end}) должен быть больше начала (t_start={t_start})."
                )
            if dt <= 0:
                raise ValueError("Шаг dt должен быть положительным.")

            noise_sigma = float(self.params["noise_sigma"])
            if noise_sigma < 0:
                raise ValueError("σ шума не может быть отрицательной.")
            seed = int(self.params["random_seed"])

            rk4 = RungeKutta4Method()
            t, sol = rk4.solve(
                model.equations_of_motion, initial_conditions, (t_start, t_end), dt
            )
            x_th = sol[:, 0].copy()
            y_th = sol[:, 1].copy()

            x_exp, y_exp = one_experimental_trajectory(x_th, y_th, noise_sigma, seed)

            sigma_rx = float(self.params["sigma_r_x"])
            sigma_ry = float(self.params["sigma_r_y"])
            if sigma_rx <= 0 or sigma_ry <= 0:
                raise ValueError("Стандартные отклонения измерений σ_rx, σ_ry должны быть > 0.")
            R = np.diag([sigma_rx**2, sigma_ry**2])

            q_scale = float(self.params["q_scale"])
            if q_scale < 0:
                raise ValueError("q_scale не может быть отрицательным.")
            Q = (q_scale**2) * np.eye(4)

            p0_scale = float(self.params["p0_scale"])
            if p0_scale <= 0:
                raise ValueError("p0_scale должен быть > 0.")
            P0 = (p0_scale**2) * np.eye(4)

            x0_ekf = np.array([x0_ic, y0_ic, vx0, vy0], dtype=float)

            xs_filt = run_ekf(
                x_exp, y_exp, t, dt, model, x0_ekf, P0, Q, R
            )

            self.results = {
                "t": t,
                "x_theory": x_th,
                "y_theory": y_th,
                "x_exp": x_exp,
                "y_exp": y_exp,
                "x_filt": xs_filt[:, 0],
                "y_filt": xs_filt[:, 1],
                "dt": dt,
                "noise_sigma": noise_sigma,
            }

            self.calculationFinished.emit(self.results)

        except Exception as e:
            error_msg = f"Ошибка при расчете: {str(e)}"
            self.calculationError.emit(error_msg)
