import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from utils.model import ConstraintMotionModel


class MotionAnimator:

    def __init__(self):
        pass

    def create_animation(
        self,
        t_data,
        x_data,
        y_data,
        title="Анимация движения",
        filename="animation.gif",
        save_animation=False,
        fps=30,
    ):
        print(f"[Animator] Создание анимации: {title}")

        cx, cy = ConstraintMotionModel.constraint_curve_xy()
        x_data = np.asarray(x_data, dtype=float)
        y_data = np.asarray(y_data, dtype=float)
        finite = np.isfinite(x_data) & np.isfinite(y_data)
        if np.any(finite):
            xmin = min(float(np.min(x_data[finite])), float(np.min(cx)))
            xmax = max(float(np.max(x_data[finite])), float(np.max(cx)))
            ymin = min(float(np.min(y_data[finite])), float(np.min(cy)))
            ymax = max(float(np.max(y_data[finite])), float(np.max(cy)))
        else:
            xmin, xmax, ymin, ymax = ConstraintMotionModel.constraint_bbox()
        dx = max(xmax - xmin, 1e-6)
        dy = max(ymax - ymin, 1e-6)
        pad = 0.1 * max(dx, dy)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlabel("Координата x")
        ax.set_ylabel("Координата y")
        ax.set_title(title)

        if len(cx) > 2:
            poly = Polygon(
                np.column_stack([cx, cy]),
                closed=True,
                facecolor="lightgray",
                edgecolor="gray",
                linewidth=1.5,
                alpha=0.45,
                label="Связь f=0",
            )
            ax.add_patch(poly)

        trajectory_line, = ax.plot([], [], "b-", lw=1, alpha=0.5, label="Пройденный путь")
        current_point, = ax.plot([], [], "ro", markersize=8, label="Объект")
        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=12)

        legend_drawn = [False]

        def init():
            trajectory_line.set_data([], [])
            current_point.set_data([], [])
            time_text.set_text("")
            return trajectory_line, current_point, time_text

        def update(frame):
            frame = min(frame, len(x_data) - 1)
            trajectory_line.set_data(x_data[: frame + 1], y_data[: frame + 1])
            current_point.set_data([x_data[frame]], [y_data[frame]])
            time_text.set_text(f"Время: {t_data[frame]:.2f} с")
            if not legend_drawn[0]:
                ax.legend(loc="upper right")
                legend_drawn[0] = True
            return trajectory_line, current_point, time_text

        total_points = len(t_data)
        max_frames = 500
        if total_points > max_frames:
            step = total_points // max_frames
            frame_indices = range(0, total_points, step)
        else:
            frame_indices = range(total_points)

        interval_ms = max(1, int(1000 / fps))

        anim = FuncAnimation(
            fig,
            update,
            frames=frame_indices,
            init_func=init,
            blit=True,
            interval=interval_ms,
            repeat=True,
        )

        plt.tight_layout()
        plt.show()

        if save_animation:
            print(f"[Animator] Сохранение анимации в '{filename}'...")
            try:
                anim.save(filename, writer="pillow", fps=fps)
                print(f"[Animator] Анимация сохранена в '{filename}'")
            except Exception as e:
                print(f"[Animator] Ошибка при сохранении анимации: {e}")
                print("[Animator] Убедитесь, что установлен необходимый writer (например, Pillow для GIF)")

        return anim
