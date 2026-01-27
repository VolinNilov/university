import numpy as np

class PendulumMotionModel:
    """
    Модель движения шарика по дуге окружности под действием силы тяжести
    """
    def __init__(self, radius, gravity=9.81, friction_coeff=0.1, mass=1.0):

        self.R = radius
        self.g = gravity
        self.b = friction_coeff
        self.m = mass

    def equations_of_motion(self, t, state):
        """
        Вычисляет производные d/dt [x, y, vx, vy].

        Args:
            t (float): Время.
            state (np.array): Текущее состояние [x, y, vx, vy].

        Returns:
            np.array: Производные [vx, vy, ax, ay].
        """
        
        x, y, vx, vy = state

        # Подавляем предупреждения NumPy при overflow/invalid (неустойчивое решение)
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            # Вычисление множителя Лагранжа lambda (см. лекцию стр. 9)
            numerator = self.m * (vx**2 + vy**2) - self.m * self.g * y
            denominator = self.R**2
            if abs(denominator) > 1e-10:
                _lambda = numerator / denominator
            else:
                _lambda = 0.0
            ax = (-_lambda * x - self.b * vx) / self.m
            ay = (-self.m * self.g - _lambda * y - self.b * vy) / self.m

        return np.array([vx, vy, ax, ay])
