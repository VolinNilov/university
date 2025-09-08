import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class MotionAnimator:
    """Класс для создания анимаций движения"""

    def __init__(self):
        pass

    def create_animation(
        self, 
        model, 
        t_data, 
        x_data, 
        y_data, 
        title="Анимация движения",
        filename="animation.gif", 
        save_animation=False, 
        fps=30
    ):
        """
        Создает и (опционально) сохраняет анимацию движения объекта.

        Args:
            model (PendulumMotionModel): Модель, использованная для расчета (для получения радиуса).
            t_data (np.array): Массив времени.
            x_data (np.array): Массив координат x.
            y_data (np.array): Массив координат y.
            title (str): Заголовок анимации.
            filename (str): Имя файла для сохранения анимации.
            save_animation (bool): Флаг для сохранения анимации в файл.
            fps (int): Кадры в секунду для сохраняемой анимации.
        """

        print(f"[Animator] Создание анимации: {title}")

        # Создаем фигуру и оси
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-model.R * 1.2, model.R * 1.2)
        ax.set_ylim(-model.R * 1.2, model.R * 1.2)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('Координата x')
        ax.set_ylabel('Координата y')
        ax.set_title(title)

        # Рисуем ограничение (окружность)
        circle = plt.Circle((0, 0), model.R, color='lightgray', fill=False, linewidth=2, label='Траектория')
        ax.add_patch(circle)

        # Инициализируем элементы анимации
        # Линия для траектории
        trajectory_line, = ax.plot([], [], 'b-', lw=1, alpha=0.5, label='Пройденный путь')
        # Текущая позиция объекта
        current_point, = ax.plot([], [], 'ro', markersize=8, label='Объект')
        # Временная метка
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

        # Используем список для хранения флага легенды
        legend_drawn = [False]

        def init():
            """Инициализатор анимации (вызывается один раз в начале)"""

            trajectory_line.set_data([], [])
            current_point.set_data([], [])
            time_text.set_text('')
            return trajectory_line, current_point, time_text

        def update(frame):
            """Функция обновления для каждого кадра анимации"""
            
            # Ограничиваем frame, чтобы не выйти за границы массива
            frame = min(frame, len(x_data) - 1)
            
            # Обновляем траекторию (все точки от начала до текущего кадра)
            trajectory_line.set_data(x_data[:frame+1], y_data[:frame+1])
            
            # Обновляем текущую позицию объекта
            current_point.set_data([x_data[frame]], [y_data[frame]])
            
            # Обновляем текст с временем
            time_text.set_text(f'Время: {t_data[frame]:.2f} с')
            
            # Отображаем легенду только один раз
            if not legend_drawn[0]:
                ax.legend(loc='upper right')
                legend_drawn[0] = True
                
            return trajectory_line, current_point, time_text

        # Создаем анимацию
        # Ограничиваем количество кадров для производительности
        total_points = len(t_data)
        max_frames = 500
        if total_points > max_frames:
            step = total_points // max_frames
            frame_indices = range(0, total_points, step)
        else:
            frame_indices = range(total_points)
        
        interval_ms = max(1, int(1000 / fps)) # Интервал в миллисекундах
        
        anim = FuncAnimation(
            fig, 
            update, 
            frames=frame_indices,
            init_func=init,
            blit=True,
            interval=interval_ms,
            repeat=True # Зациклить анимацию
        )

        # Отображаем анимацию
        plt.tight_layout()
        plt.show()

        # Сохраняем анимацию, если это требуется
        if save_animation:
            print(f"[Animator] Сохранение анимации в '{filename}'...")
            try:
                # Для сохранения в GIF нужен Pillow (pip install Pillow)
                anim.save(filename, writer='pillow', fps=fps)
                print(f"[Animator] Анимация сохранена в '{filename}'")
            except Exception as e:
                print(f"[Animator] Ошибка при сохранении анимации: {e}")
                print("[Animator] Убедитесь, что установлен необходимый writer (например, Pillow для GIF)")

        return anim # Возвращаем объект анимации на случай, если он понадобится
