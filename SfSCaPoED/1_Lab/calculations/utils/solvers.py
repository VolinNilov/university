import numpy as np
from abc import ABC, abstractmethod

# --- Интерфейс для численных решателей ---
class ODESolver(ABC):
    """Абстрактный базовый класс для численных решателей ОДУ."""
    @abstractmethod
    def solve(self, equations_of_motion, initial_conditions, t_span, dt):
        """
        Решает систему ОДУ.

        Args:
            equations_of_motion (callable): Функция, вычисляющая производные.
                                            Принимает (t, state) -> derivatives.
            initial_conditions (list or np.array): Начальные условия [x0, y0, vx0, vy0].
            t_span (tuple): Интервал интегрирования (t_start, t_end).
            dt (float): Шаг по времени.

        Returns:
            tuple: (time_points, solution_array)
                   time_points (np.array): Массив моментов времени.
                   solution_array (np.array): Массив решений [x, y, vx, vy] на каждом шаге.
        """
        pass

# --- Конкретные реализации решателей ---
class EulerMethod(ODESolver):
    """Реализация метода Эйлера."""
    def solve(self, equations_of_motion, initial_conditions, t_span, dt):
        t_start, t_end = t_span
        num_steps = int((t_end - t_start) / dt) + 1
        t = np.linspace(t_start, t_end, num_steps)
        
        sol = np.zeros((num_steps, len(initial_conditions)))
        sol[0] = np.array(initial_conditions)

        for i in range(1, num_steps):
            derivatives = equations_of_motion(t[i-1], sol[i-1])
            sol[i] = sol[i-1] + dt * derivatives
            
        return t, sol

class HeunMethod(ODESolver):
    """Реализация метода Хьюна (предиктор-корректор)."""
    def solve(self, equations_of_motion, initial_conditions, t_span, dt):
        t_start, t_end = t_span
        num_steps = int((t_end - t_start) / dt) + 1
        t = np.linspace(t_start, t_end, num_steps)
        
        sol = np.zeros((num_steps, len(initial_conditions)))
        sol[0] = np.array(initial_conditions)

        for i in range(1, num_steps):
            # Предиктор (метод Эйлера)
            derivatives_n = equations_of_motion(t[i-1], sol[i-1])
            y_pred = sol[i-1] + dt * derivatives_n
            
            # Вычисляем производные в предсказанной точке
            derivatives_np1 = equations_of_motion(t[i], y_pred)
            
            # Корректор (усреднение)
            sol[i] = sol[i-1] + 0.5 * dt * (derivatives_n + derivatives_np1)
            
        return t, sol

class RungeKutta4Method(ODESolver):
    """Реализация метода Рунге-Кутты 4-го порядка."""
    def solve(self, equations_of_motion, initial_conditions, t_span, dt):
        t_start, t_end = t_span
        num_steps = int((t_end - t_start) / dt) + 1
        t = np.linspace(t_start, t_end, num_steps)
        
        sol = np.zeros((num_steps, len(initial_conditions)))
        sol[0] = np.array(initial_conditions)

        for i in range(1, num_steps):
            tn = t[i-1]
            yn = sol[i-1]
            
            # Вычисление коэффициентов k1, k2, k3, k4
            k1 = dt * equations_of_motion(tn, yn)
            k2 = dt * equations_of_motion(tn + dt/2, yn + k1/2)
            k3 = dt * equations_of_motion(tn + dt/2, yn + k2/2)
            k4 = dt * equations_of_motion(tn + dt, yn + k3)
            
            # Обновление решения
            sol[i] = yn + (k1 + 2*k2 + 2*k3 + k4) / 6
            
        return t, sol
