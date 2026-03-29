import numpy as np
from abc import ABC, abstractmethod


class ODESolver(ABC):

    @abstractmethod
    def solve(self, equations_of_motion, initial_conditions, t_span, dt):
        pass


class EulerMethod(ODESolver):

    def solve(self, equations_of_motion, initial_conditions, t_span, dt):
        t_start, t_end = t_span
        if t_end <= t_start or dt <= 0:
            raise ValueError("t_end > t_start и dt > 0 обязательны для интегрирования.")
        num_steps = max(2, int((t_end - t_start) / dt) + 1)
        t = np.linspace(t_start, t_end, num_steps)

        sol = np.zeros((num_steps, len(initial_conditions)))
        sol[0] = np.array(initial_conditions)

        for i in range(1, num_steps):
            derivatives = equations_of_motion(t[i - 1], sol[i - 1])
            sol[i] = sol[i - 1] + dt * derivatives

        return t, sol


class HeunMethod(ODESolver):

    def solve(self, equations_of_motion, initial_conditions, t_span, dt):
        t_start, t_end = t_span
        if t_end <= t_start or dt <= 0:
            raise ValueError("t_end > t_start и dt > 0 обязательны для интегрирования.")
        num_steps = max(2, int((t_end - t_start) / dt) + 1)
        t = np.linspace(t_start, t_end, num_steps)

        sol = np.zeros((num_steps, len(initial_conditions)))
        sol[0] = np.array(initial_conditions)

        for i in range(1, num_steps):
            derivatives_n = equations_of_motion(t[i - 1], sol[i - 1])
            y_pred = sol[i - 1] + dt * derivatives_n
            derivatives_np1 = equations_of_motion(t[i], y_pred)
            sol[i] = sol[i - 1] + 0.5 * dt * (derivatives_n + derivatives_np1)

        return t, sol


class RungeKutta4Method(ODESolver):

    def solve(self, equations_of_motion, initial_conditions, t_span, dt):
        t_start, t_end = t_span
        if t_end <= t_start or dt <= 0:
            raise ValueError("t_end > t_start и dt > 0 обязательны для интегрирования.")
        num_steps = max(2, int((t_end - t_start) / dt) + 1)
        t = np.linspace(t_start, t_end, num_steps)

        sol = np.zeros((num_steps, len(initial_conditions)))
        sol[0] = np.array(initial_conditions)

        for i in range(1, num_steps):
            tn = t[i - 1]
            yn = sol[i - 1]
            k1 = dt * equations_of_motion(tn, yn)
            k2 = dt * equations_of_motion(tn + dt / 2, yn + k1 / 2)
            k3 = dt * equations_of_motion(tn + dt / 2, yn + k2 / 2)
            k4 = dt * equations_of_motion(tn + dt, yn + k3)
            sol[i] = yn + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return t, sol
