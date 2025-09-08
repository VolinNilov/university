import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.solvers import EulerMethod, HeunMethod, RungeKutta4Method
from utils.model import PendulumMotionModel
from utils.simulator import MotionSimulator
from PyQt6.QtCore import QObject, pyqtSignal

class AppModel(QObject):
    """
    Модель для MVC-приложения. Управляет данными и логикой расчетов.
    """

    # Сигналы для уведомления View/Controller об изменениях
    calculationFinished = pyqtSignal(dict) # Передает результаты расчетов
    calculationStarted = pyqtSignal()
    calculationError = pyqtSignal(str)
    animationDataReady = pyqtSignal(dict) # Передает данные для анимации

    def __init__(self):
        super().__init__()
        self.simulator = None
        self.results = {}

    def set_parameters(self, params):
        """Устанавливает параметры модели из словаря"""

        self.params = params

    def run_calculations(self):
        """Запускает расчеты на основе установленных параметров"""
        
        try:
            self.calculationStarted.emit()
            
            # 1. Определяем модель движения
            model = PendulumMotionModel(
                radius=self.params['radius'],
                gravity=self.params['gravity'],
                friction_coeff=self.params['friction_coeff'],
                mass=self.params['mass']
            )

            # 2. Задаем начальные условия
            R = model.R
            angle_rad = self.params['angle_rad']
            x0 = R * np.sin(angle_rad)
            y0 = -R * np.cos(angle_rad) # Отрицательный, как в примере
            vx0 = self.params['vx0']
            vy0 = self.params['vy0']
            initial_conditions = [x0, y0, vx0, vy0]

            # 3. Задаем параметры интегрирования
            t_span = (self.params['t_start'], self.params['t_end'])
            dt_coarse = self.params['dt_coarse']
            dt_fine = self.params['dt_fine']

            # 4. Создаем решатели
            euler_solver = EulerMethod()
            heun_solver = HeunMethod()
            rk4_solver = RungeKutta4Method()

            # 5. Создаем "виртуальный" симулятор только для расчетов
            # (не будем сохранять в файлы из UI)
            self.simulator = MotionSimulator(model, output_dir_name="data") # output_dir не используется в этом контексте

            # 6. Запускаем симуляции
            print("[AppModel] Запуск расчетов...")
            
            self.simulator.run_simulation(euler_solver, initial_conditions, t_span, dt_coarse, f'Euler (dt={dt_coarse})')
            self.simulator.run_simulation(heun_solver, initial_conditions, t_span, dt_coarse, f'Heun (dt={dt_coarse})')
            self.simulator.run_simulation(rk4_solver, initial_conditions, t_span, dt_coarse, f'RK4 (dt={dt_coarse})')
            self.simulator.run_simulation(rk4_solver, initial_conditions, t_span, dt_fine, f'RK4_ref (dt={dt_fine})')

            self.results = self.simulator.results
            self.calculationFinished.emit(self.results)
            
            # Подготовить данные для анимации (например, эталонную траекторию)
            # Можно выбрать любую, например, RK4 с мелким шагом
            anim_key = 'RK4_ref (dt=0.005)'
            if anim_key in self.results:
                self.animationDataReady.emit(self.results[anim_key])
            else:
                # Если эталонной нет, возьмем первую попавшуюся
                first_key = next(iter(self.results))
                self.animationDataReady.emit(self.results[first_key])

        except Exception as e:
            error_msg = f"Ошибка при расчете: {str(e)}"
            print(f"[AppModel] {error_msg}")
            self.calculationError.emit(error_msg)
