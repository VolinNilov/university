import numpy as np


class ConstraintMotionModel:

    def __init__(self, gravity=9.81, friction_coeff=0.1, mass=1.0):
        self.g = gravity
        self.b = friction_coeff
        self.m = mass

    @staticmethod
    def f(x, y):
        return 0.5 * x * x + y * y + 0.5 * x * y - 4.0 * y

    @staticmethod
    def grad_f(x, y):
        fx = x + 0.5 * y
        fy = 0.5 * x + 2.0 * y - 4.0
        return fx, fy

    def equations_of_motion(self, t, state):
        x, y, vx, vy = state
        fx, fy = self.grad_f(x, y)
        H = vx * vx + vx * vy + 2.0 * vy * vy
        gsq = fx * fx + fy * fy
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            if abs(gsq) > 1e-14:
                lam = self.m * (self.g * fy - H) / gsq
            else:
                lam = 0.0
            ax = (lam * fx - self.b * vx) / self.m
            ay = (-self.m * self.g + lam * fy - self.b * vy) / self.m

        return np.array([vx, vy, ax, ay])

    @staticmethod
    def constraint_curve_xy(num=600):
        y_max = 32.0 / 7.0
        ys_grid = np.linspace(0.0, y_max, max(4, num // 2))
        left_branch = []
        for y in ys_grid:
            disc = 8.0 * y - 1.75 * y * y
            if disc < 0.0:
                continue
            s = np.sqrt(disc)
            left_branch.append((-0.5 * y - s, y))
        right_branch = []
        for y in reversed(ys_grid):
            disc = 8.0 * y - 1.75 * y * y
            if disc < 0.0:
                continue
            s = np.sqrt(disc)
            right_branch.append((-0.5 * y + s, y))
        loop = left_branch + right_branch
        if not loop:
            return np.array([0.0]), np.array([0.0])
        p0, p1 = loop[0], loop[-1]
        if (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 > 1e-10:
            loop = loop + [loop[0]]
        xs = np.array([p[0] for p in loop], dtype=float)
        ys = np.array([p[1] for p in loop], dtype=float)
        return xs, ys

    @staticmethod
    def constraint_bbox(margin_ratio=0.12):
        xc, yc = ConstraintMotionModel.constraint_curve_xy()
        xmin, xmax = float(np.min(xc)), float(np.max(xc))
        ymin, ymax = float(np.min(yc)), float(np.max(yc))
        dx = max(xmax - xmin, 1e-6)
        dy = max(ymax - ymin, 1e-6)
        pad_x = dx * margin_ratio
        pad_y = dy * margin_ratio
        return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y
