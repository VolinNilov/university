import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.solvers import EulerMethod, HeunMethod, RungeKutta4Method
from utils.model import ConstraintMotionModel
from utils.simulator import MotionSimulator
from PyQt6.QtCore import QObject, pyqtSignal

_F_TOL = 1e-3
_V_TOL = 1e-3


class AppModel(QObject):
    """
    Модель для MVC-приложения. Управляет данными и логикой расчетов.
    """

    calculationFinished = pyqtSignal(dict)
    calculationStarted = pyqtSignal()
    calculationError = pyqtSignal(str)
    animationDataReady = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.simulator = None
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
            dt_coarse = self.params["dt_coarse"]
            dt_fine = self.params["dt_fine"]
            if t_end <= t_start:
                raise ValueError(
                    f"Конец интервала (t_end={t_end}) должен быть больше начала (t_start={t_start})."
                )
            if dt_coarse <= 0 or dt_fine <= 0:
                raise ValueError("Шаги интегрирования dt должны быть положительными.")
            t_span = (t_start, t_end)

            euler_solver = EulerMethod()
            heun_solver = HeunMethod()
            rk4_solver = RungeKutta4Method()

            self.simulator = MotionSimulator(model, output_dir_name="data")

            print("[AppModel] Запуск расчетов...")

            self.simulator.run_simulation(
                euler_solver, initial_conditions, t_span, dt_coarse, f"Euler (dt={dt_coarse})"
            )
            self.simulator.run_simulation(
                heun_solver, initial_conditions, t_span, dt_coarse, f"Heun (dt={dt_coarse})"
            )
            self.simulator.run_simulation(
                rk4_solver, initial_conditions, t_span, dt_coarse, f"RK4 (dt={dt_coarse})"
            )
            self.simulator.run_simulation(
                rk4_solver, initial_conditions, t_span, dt_fine, f"RK4_ref (dt={dt_fine})"
            )

            self.results = self.simulator.results
            self.calculationFinished.emit(self.results)

            anim_key = f"RK4_ref (dt={dt_fine})"
            if anim_key in self.results:
                payload = dict(self.results[anim_key])
                payload["_bbox"] = self._trajectory_bbox(self.results[anim_key])
                self.animationDataReady.emit(payload)
            else:
                first_key = next(iter(self.results))
                payload = dict(self.results[first_key])
                payload["_bbox"] = self._trajectory_bbox(self.results[first_key])
                self.animationDataReady.emit(payload)

        except Exception as e:
            error_msg = f"Ошибка при расчете: {str(e)}"
            print(f"[AppModel] {error_msg}")
            self.calculationError.emit(error_msg)

    @staticmethod
    def _trajectory_bbox(data, margin_ratio=0.08):
        x = np.asarray(data["x"], dtype=float)
        y = np.asarray(data["y"], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            return ConstraintMotionModel.constraint_bbox()
        x, y = x[finite], y[finite]
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
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
