import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.solvers import RungeKutta4Method
from utils.model import ConstraintMotionModel
from utils.experiment import ensemble_experimental, mean_and_ci_student_t
from PyQt6.QtCore import QObject, pyqtSignal

_F_TOL = 1e-3
_V_TOL = 1e-3
_ALPHA = 0.05


class AppModel(QObject):

    calculationFinished = pyqtSignal(dict)
    calculationStarted = pyqtSignal()
    calculationError = pyqtSignal(str)
    animationDataReady = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.results = {}

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

            x0 = self.params["x0"]
            y0 = self.params["y0"]
            vx0 = self.params["vx0"]
            vy0 = self.params["vy0"]

            f0 = ConstraintMotionModel.f(x0, y0)
            if abs(f0) > _F_TOL:
                raise ValueError(
                    f"Точка ({x0}, {y0}) не на связи: f ≈ {f0:.6g} (допуск |f| ≤ {_F_TOL})."
                )
            gfx, gfy = ConstraintMotionModel.grad_f(x0, y0)
            fdot = gfx * vx0 + gfy * vy0
            if abs(fdot) > _V_TOL:
                raise ValueError(
                    f"Скорость не касательна к связи: ∇f·v ≈ {fdot:.6g} (допуск ≤ {_V_TOL})."
                )

            initial_conditions = [x0, y0, vx0, vy0]

            t_start = self.params["t_start"]
            t_end = self.params["t_end"]
            dt_fine = self.params["dt_fine"]
            if t_end <= t_start:
                raise ValueError(
                    f"Конец интервала (t_end={t_end}) должен быть больше начала (t_start={t_start})."
                )
            if dt_fine <= 0:
                raise ValueError("Шаг интегрирования dt должен быть положительным.")
            t_span = (t_start, t_end)

            n_runs = int(self.params["n_runs"])
            if n_runs < 2:
                raise ValueError("Число траекторий N должно быть не меньше 2.")

            noise_sigma = float(self.params["noise_sigma"])
            if noise_sigma < 0:
                raise ValueError("σ шума не может быть отрицательной.")
            seed_base = int(self.params["random_seed"])

            rk4 = RungeKutta4Method()
            t, sol = rk4.solve(model.equations_of_motion, initial_conditions, t_span, dt_fine)
            x_th = sol[:, 0].copy()
            y_th = sol[:, 1].copy()

            x_exp, y_exp = ensemble_experimental(x_th, y_th, n_runs, noise_sigma, seed_base)

            x_mean, x_lo, x_hi = mean_and_ci_student_t(x_exp, alpha=_ALPHA)
            y_mean, y_lo, y_hi = mean_and_ci_student_t(y_exp, alpha=_ALPHA)

            self.results = {
                "lab2": True,
                "dt_fine": dt_fine,
                "t": t,
                "x_theory": x_th,
                "y_theory": y_th,
                "x_exp": x_exp,
                "y_exp": y_exp,
                "x_mean": x_mean,
                "y_mean": y_mean,
                "x_ci_lo": x_lo,
                "x_ci_hi": x_hi,
                "y_ci_lo": y_lo,
                "y_ci_hi": y_hi,
                "N": n_runs,
                "noise_sigma": noise_sigma,
                "alpha": _ALPHA,
            }

            print("[AppModel] ЛР2: теория + ансамбль готовы.")
            self.calculationFinished.emit(self.results)

            bbox = self._bbox_for_xy(x_th, y_th, x_exp, y_exp)
            anim = {
                "t": t,
                "x": x_mean,
                "y": y_mean,
                "_bbox": bbox,
            }
            self.animationDataReady.emit(anim)

        except Exception as e:
            error_msg = f"Ошибка при расчете: {str(e)}"
            print(f"[AppModel] {error_msg}")
            self.calculationError.emit(error_msg)

    @staticmethod
    def _bbox_for_xy(x_th, y_th, x_exp, y_exp, margin_ratio=0.08):
        xs = np.concatenate([x_th.ravel(), x_exp.ravel()])
        ys = np.concatenate([y_th.ravel(), y_exp.ravel()])
        finite = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(finite):
            return ConstraintMotionModel.constraint_bbox()
        xs, ys = xs[finite], ys[finite]
        xmin, xmax = float(np.min(xs)), float(np.max(xs))
        ymin, ymax = float(np.min(ys)), float(np.max(ys))
        cx, cy = ConstraintMotionModel.constraint_curve_xy()
        xmin = min(xmin, float(np.min(cx)))
        xmax = max(xmax, float(np.max(cx)))
        ymin = min(ymin, float(np.min(cy)))
        ymax = max(ymax, float(np.max(cy)))
        dx = max(xmax - xmin, 1e-6)
        dy = max(ymax - ymin, 1e-6)
        pad_x = dx * margin_ratio
        pad_y = dy * margin_ratio
        return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y
