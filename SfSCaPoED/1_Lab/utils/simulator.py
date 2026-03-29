import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

class MotionSimulator:
    """Класс для запуска симуляций с разными решателями и сравнения результатов"""
    
    def __init__(self, model, output_dir_name="data"):
        """
        Args:
            model: Модель движения (например, ConstraintMotionModel).
            output_dir_name (str): Имя директории для сохранения результатов (относительно скрипта).
        """

        self.model = model
        self.results = {}
        
        # Получаем путь к директории, где находится main.py
        script_dir = Path(__file__).parent.parent.resolve()
        self.output_dir = script_dir / output_dir_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Simulator] Результаты будут сохранены в: {self.output_dir}")

    def run_simulation(self, solver, initial_conditions, t_span, dt, label):
        """
        Запускает симуляцию с заданным решателем.

        Args:
            solver (ODESolver): Экземпляр решателя.
            initial_conditions (list): Начальные условия [x0, y0, vx0, vy0].
            t_span (tuple): Интервал времени (t_start, t_end).
            dt (float): Шаг времени.
            label (str): Метка для идентификации результата.
        """

        t, sol = solver.solve(self.model.equations_of_motion, initial_conditions, t_span, dt)
        x, y, vx, vy = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]
        self.results[label] = {'t': t, 'x': x, 'y': y, 'vx': vx, 'vy': vy}
        print(f"[Simulator] Симуляция '{label}' завершена. Шаг: {dt}")

    def plot_lab1_graphs(self, filename_prefix="lab1_graphs"):
        """
        Строит все требуемые графики ЛР1: x(t), y(t), vx(t), vy(t), y(x).
        """

        if not self.results:
            print("[Simulator] Нет результатов для отображения.")
            return

        # Создаем фигуру с 3x2 подграфиками
        fig, axs = plt.subplots(3, 2, figsize=(18, 12))
        fig.suptitle('Лабораторная работа 1: Графики движения', fontsize=16)

        # 1. Координата x от времени
        ax = axs[0, 0]
        for label, data in self.results.items():
            ax.plot(data['t'], data['x'], label=label)
        ax.set_xlabel('Время t')
        ax.set_ylabel('Координата x')
        ax.set_title('Координата x от времени')
        ax.legend()
        ax.grid(True)

        # 2. Координата y от времени
        ax = axs[0, 1]
        for label, data in self.results.items():
            ax.plot(data['t'], data['y'], label=label)
        ax.set_xlabel('Время t')
        ax.set_ylabel('Координата y')
        ax.set_title('Координата y от времени')
        ax.legend()
        ax.grid(True)

        # 3. Проекция скорости vx от времени
        ax = axs[1, 0]
        for label, data in self.results.items():
            ax.plot(data['t'], data['vx'], label=label)
        ax.set_xlabel('Время t')
        ax.set_ylabel('Скорость vx')
        ax.set_title('Проекция скорости vx от времени')
        ax.legend()
        ax.grid(True)

        # 4. Проекция скорости vy от времени
        ax = axs[1, 1]
        for label, data in self.results.items():
            ax.plot(data['t'], data['vy'], label=label)
        ax.set_xlabel('Время t')
        ax.set_ylabel('Скорость vy')
        ax.set_title('Проекция скорости vy от времени')
        ax.legend()
        ax.grid(True)

        # 5. Траектория y от x
        ax = axs[2, 0]
        for label, data in self.results.items():
            ax.plot(data['x'], data['y'], label=label)
        ax.set_xlabel('Координата x')
        ax.set_ylabel('Координата y')
        ax.set_title('Траектория (y от x)')
        ax.legend()
        ax.grid(True)
        # ax.set_aspect('equal', adjustable='box') # Опционально, для окружности

        # 6. Траектория x от y (как в примере лекции)
        ax = axs[2, 1]
        for label, data in self.results.items():
            ax.plot(data['x'], data['y'], label=label)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Траектория (x от y)')
        ax.legend()
        ax.grid(True)
        # ax.set_aspect('equal', adjustable='box')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_filename = self.output_dir / f"{filename_prefix}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"[Simulator] Графики ЛР1 сохранены в '{plot_filename}'")
        plt.show()

    def save_results_to_csv(self):
        """Сохраняет все результаты в отдельные CSV файлы"""

        for label, data in self.results.items():
            # Очищаем имя файла от недопустимых символов
            safe_label = "".join(c for c in label if c.isalnum() or c in (' ','.','_','-')).rstrip()
            safe_label = safe_label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_")
            filename = self.output_dir / f"results_{safe_label}.csv"
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Time', 'x', 'y', 'vx', 'vy'])
                for i in range(len(data['t'])):
                    writer.writerow([data['t'][i], data['x'][i], data['y'][i], data['vx'][i], data['vy'][i]])
            print(f"[Simulator] Данные '{label}' сохранены в '{filename}'")

    def print_comparison(self, ref_label, *other_labels):
        """
        Печатает сравнение результатов относительно эталонного решения.
        Сравнивает конечные координаты.
        """
        
        if ref_label not in self.results:
            print(f"[Simulator] Эталонное решение '{ref_label}' не найдено.")
            return

        ref_data = self.results[ref_label]
        ref_x_end, ref_y_end = ref_data['x'][-1], ref_data['y'][-1]
        print(f"\n[Simulator] Сравнение с '{ref_label}' (x_end={ref_x_end:.6f}, y_end={ref_y_end:.6f}):")

        for label in other_labels:
            if label in self.results:
                data = self.results[label]
                x_end, y_end = data['x'][-1], data['y'][-1]
                diff_x = abs(x_end - ref_x_end)
                diff_y = abs(y_end - ref_y_end)
                print(f"  {label}: Δx={diff_x:.5e}, Δy={diff_y:.5e}")
            else:
                print(f"  Решение '{label}' не найдено.")
