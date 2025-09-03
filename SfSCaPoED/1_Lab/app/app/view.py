import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGroupBox, QFormLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QSplitter, QSizePolicy,
    QDoubleSpinBox, QSpinBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

class InputGroupBox(QGroupBox):
    """Группа для ввода параметров."""
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        layout = QFormLayout()
        self.setLayout(layout)

    def addRow(self, label_text, widget):
        self.layout().addRow(QLabel(label_text), widget)

class PlotCanvas(FigureCanvas):
    """Холст для matplotlib."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

class AnimationWidget(QWidget):
    """Виджет для анимации."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("background-color: white; border: 1px solid gray;")
        
        self.x_data = []
        self.y_data = []
        self.t_data = []
        self.current_index = 0
        self.radius = 1.0
        
        # Таймер для анимации
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.animation_speed = 50 # мс

    def set_data(self, data_dict, radius=1.0):
        """Устанавливает данные для анимации."""
        print(f"[AnimationWidget] set_data вызван. radius={radius}")
        self.x_data = data_dict.get('x', [])
        self.y_data = data_dict.get('y', [])
        self.t_data = data_dict.get('t', [])
        print(f"[AnimationWidget] Получены данные: x({len(self.x_data)}), y({len(self.y_data)}), t({len(self.t_data)})")
        self.current_index = 0
        self.radius = radius
        print(f"[AnimationWidget] self.radius установлен на {self.radius}")
        self.update() # Перерисовать первый кадр
        print("[AnimationWidget] update() вызван")

    def start_animation(self):
        if len(self.x_data) > 1:
            self.current_index = 0
            self.timer.start(self.animation_speed)

    def stop_animation(self):
        self.timer.stop()

    def update_animation(self):
        if self.current_index < len(self.x_data) - 1:
            self.current_index += 1
            self.update()  # Обновляем отображение
        else:
            self.timer.stop() # Остановить в конце

    def paintEvent(self, event):
        print(f"[AnimationWidget] paintEvent вызван. current_index={self.current_index}, data_len_x={len(self.x_data) if hasattr(self, 'x_data') and isinstance(self.x_data, (list, np.ndarray)) else 'N/A'}")
        # Исправленная проверка на пустоту данных
        if (not isinstance(self.x_data, (list, np.ndarray)) or len(self.x_data) == 0 or
            not isinstance(self.y_data, (list, np.ndarray)) or len(self.y_data) == 0):
            print("[AnimationWidget] paintEvent: Нет данных для отрисовки")
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Размеры виджета
        width = self.width()
        height = self.height()
        margin = 20
        
        # --- Отладочный вывод масштаба и смещения ---
        print(f"[paintEvent Debug] Widget size: {width}x{height}, margin: {margin}")
        print(f"[paintEvent Debug] self.radius: {self.radius}")
        
        # Масштабирование (простое, можно улучшить)
        max_coord = self.radius * 1.2
        if max_coord == 0: max_coord = 1
        scale = min((width - 2*margin), (height - 2*margin)) / (2 * max_coord)
        
        # --- Отладочный вывод scale ---
        print(f"[paintEvent Debug] max_coord: {max_coord}, scale: {scale}")
        
        # Центр координат (внизу посередине, как в примере лекции)
        center_x = width // 2
        center_y = height - margin - 50 # Немного выше нижней границы
        
        # --- Отладочный вывод центра ---
        print(f"[paintEvent Debug] center: ({center_x}, {center_y})")

        # Рисуем ограничение (например, окружность)
        pen = QPen(QColor("lightgray"), 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        circle_top_left_x = int(center_x - self.radius * scale)
        circle_top_left_y = int(center_y - self.radius * scale)
        circle_width = int(2 * self.radius * scale)
        circle_height = int(2 * self.radius * scale)
        # --- Отладочный вывод окружности ---
        print(f"[paintEvent Debug] Circle rect: ({circle_top_left_x}, {circle_top_left_y}, {circle_width}, {circle_height})")
        painter.drawEllipse(circle_top_left_x, circle_top_left_y, circle_width, circle_height)

        # Рисуем траекторию
        pen = QPen(QColor("lightblue"), 1, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        # Ограничиваем количество рисуемых сегментов для отладки, если нужно
        num_segments_to_draw = min(self.current_index + 1, 10) # Например, только первые 10
        num_segments_to_draw = self.current_index + 1
        for i in range(1, min(num_segments_to_draw, len(self.x_data))):
            x1 = center_x + int(self.x_data[i-1] * scale)
            y1 = center_y - int(self.y_data[i-1] * scale) # Инвертируем Y
            x2 = center_x + int(self.x_data[i] * scale)
            y2 = center_y - int(self.y_data[i] * scale)
            # --- Отладочный вывод первых нескольких точек траектории ---
            if i < 5:
                print(f"[paintEvent Debug] Trajectory segment {i}: ({x1}, {y1}) -> ({x2}, {y2})")
            painter.drawLine(x1, y1, x2, y2)

        # Рисуем объект (шарик)
        if 0 <= self.current_index < len(self.x_data):
            obj_x_unscaled = self.x_data[self.current_index]
            obj_y_unscaled = self.y_data[self.current_index]
            obj_x = center_x + int(obj_x_unscaled * scale)
            obj_y = center_y - int(obj_y_unscaled * scale) # Инвертируем Y
            
            # --- Отладочный вывод координат объекта ---
            print(f"[paintEvent Debug] Object raw coords: ({obj_x_unscaled}, {obj_y_unscaled})")
            print(f"[paintEvent Debug] Object scaled coords: ({obj_x}, {obj_y})")
            
            # Проверка, находится ли объект в пределах виджета (грубая)
            if not (0 <= obj_x <= width and 0 <= obj_y <= height):
                print(f"[paintEvent Debug] WARNING: Object is likely outside widget bounds!")
            
            pen = QPen(QColor("red"), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            brush_color = QColor("red")
            painter.setBrush(brush_color)
            painter.drawEllipse(obj_x - 5, obj_y - 5, 10, 10)
            
            # Подпись с временем
            painter.setPen(QPen(QColor("black")))
            painter.setFont(QFont("Arial", 8))
            # Отображаем реальное время, если оно есть
            time_str = f"i={self.current_index}"
            if 0 <= self.current_index < len(self.t_data):
                time_str = f"t={self.t_data[self.current_index]:.3f}s"
            painter.drawText(obj_x + 10, obj_y, time_str)
            # --- Отладочный вывод текста ---
            # print(f"[paintEvent Debug] Drawing text '{time_str}' at ({obj_x + 10}, {obj_y})")

        # --- Отладочный вывод в конце ---
        print(f"[paintEvent Debug] Frame for index {self.current_index} drawn.")


class MainWindow(QMainWindow):
    """Главное окно приложения."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа 1 - Моделирование движения")
        self.setGeometry(100, 100, 1400, 900)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Левая панель (ввод данных и управление)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(350)

        # --- Группа параметров модели ---
        self.model_group = InputGroupBox("Параметры модели")
        self.le_radius = QDoubleSpinBox()
        self.le_radius.setRange(0.1, 100.0)
        self.le_radius.setSingleStep(0.1)
        self.le_radius.setValue(1.0)
        self.model_group.addRow("Радиус (R):", self.le_radius)

        self.le_gravity = QDoubleSpinBox()
        self.le_gravity.setRange(0.1, 50.0)
        self.le_gravity.setSingleStep(0.1)
        self.le_gravity.setValue(9.81)
        self.model_group.addRow("Ускорение (g):", self.le_gravity)

        self.le_friction = QDoubleSpinBox()
        self.le_friction.setRange(0.0, 10.0)
        self.le_friction.setSingleStep(0.01)
        self.le_friction.setValue(0.2)
        self.model_group.addRow("Трение (b):", self.le_friction)

        self.le_mass = QDoubleSpinBox()
        self.le_mass.setRange(0.1, 100.0)
        self.le_mass.setSingleStep(0.1)
        self.le_mass.setValue(1.0)
        self.model_group.addRow("Масса (m):", self.le_mass)
        left_layout.addWidget(self.model_group)

        # --- Группа начальных условий ---
        self.ic_group = InputGroupBox("Начальные условия")
        self.le_angle = QDoubleSpinBox()
        self.le_angle.setRange(-np.pi, np.pi)
        self.le_angle.setSingleStep(0.01)
        self.le_angle.setValue(0.1)
        self.le_angle.setDecimals(3)
        self.ic_group.addRow("Угол (рад):", self.le_angle)

        self.le_vx0 = QDoubleSpinBox()
        self.le_vx0.setRange(-100.0, 100.0)
        self.le_vx0.setSingleStep(0.1)
        self.le_vx0.setValue(0.0)
        self.ic_group.addRow("Vx0:", self.le_vx0)

        self.le_vy0 = QDoubleSpinBox()
        self.le_vy0.setRange(-100.0, 100.0)
        self.le_vy0.setSingleStep(0.1)
        self.le_vy0.setValue(0.0)
        self.ic_group.addRow("Vy0:", self.le_vy0)
        left_layout.addWidget(self.ic_group)

        # --- Группа параметров интегрирования ---
        self.int_group = InputGroupBox("Параметры интегрирования")
        self.le_t_start = QDoubleSpinBox()
        self.le_t_start.setRange(0.0, 1000.0)
        self.le_t_start.setSingleStep(0.1)
        self.le_t_start.setValue(0.0)
        self.int_group.addRow("t_start:", self.le_t_start)

        self.le_t_end = QDoubleSpinBox()
        self.le_t_end.setRange(0.1, 1000.0)
        self.le_t_end.setSingleStep(0.1)
        self.le_t_end.setValue(5.0)
        self.int_group.addRow("t_end:", self.le_t_end)

        self.le_dt_coarse = QDoubleSpinBox()
        self.le_dt_coarse.setRange(0.0001, 1.0)
        self.le_dt_coarse.setSingleStep(0.001)
        self.le_dt_coarse.setDecimals(4)
        self.le_dt_coarse.setValue(0.05)
        self.int_group.addRow("dt (грубый):", self.le_dt_coarse)

        self.le_dt_fine = QDoubleSpinBox()
        self.le_dt_fine.setRange(0.0001, 1.0)
        self.le_dt_fine.setSingleStep(0.0001)
        self.le_dt_fine.setDecimals(5)
        self.le_dt_fine.setValue(0.005)
        self.int_group.addRow("dt (мелкий):", self.le_dt_fine)
        left_layout.addWidget(self.int_group)

        # --- Кнопки управления ---
        self.btn_run = QPushButton("Запустить расчеты")
        self.btn_run.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(self.btn_run)
        
        self.btn_animate = QPushButton("Запустить анимацию")
        self.btn_animate.setEnabled(False) # Активна только после расчетов
        left_layout.addWidget(self.btn_animate)
        
        self.btn_stop_animate = QPushButton("Остановить анимацию")
        self.btn_stop_animate.setEnabled(False)
        left_layout.addWidget(self.btn_stop_animate)

        left_layout.addStretch()

        # --- Правая панель (графики и анимация) ---
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Верхняя часть - графики
        self.plot_tabs = QWidget()
        plot_layout = QVBoxLayout(self.plot_tabs)
        
        # Создаем холсты для графиков
        self.fig_canvas = FigureCanvas(plt.Figure(figsize=(10, 8)))
        plot_layout.addWidget(self.fig_canvas)
        
        # Нижняя часть - анимация
        self.animation_widget = AnimationWidget()
        # Изначально делаем её меньше
        right_splitter.addWidget(self.plot_tabs)
        right_splitter.addWidget(self.animation_widget)
        right_splitter.setSizes([600, 300]) # Примерные размеры

        # Добавляем панели в основной лейаут
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_splitter)

        # --- Хранилище для данных графиков ---
        self.plot_data = {}

    def get_parameters(self):
        """Собирает параметры из полей ввода."""
        return {
            'radius': self.le_radius.value(),
            'gravity': self.le_gravity.value(),
            'friction_coeff': self.le_friction.value(),
            'mass': self.le_mass.value(),
            'angle_rad': self.le_angle.value(),
            'vx0': self.le_vx0.value(),
            'vy0': self.le_vy0.value(),
            't_start': self.le_t_start.value(),
            't_end': self.le_t_end.value(),
            'dt_coarse': self.le_dt_coarse.value(),
            'dt_fine': self.le_dt_fine.value(),
        }

    def set_parameters(self, params):
        """Устанавливает параметры в поля ввода."""
        self.le_radius.setValue(params['radius'])
        self.le_gravity.setValue(params['gravity'])
        self.le_friction.setValue(params['friction_coeff'])
        self.le_mass.setValue(params['mass'])
        self.le_angle.setValue(params['angle_rad'])
        self.le_vx0.setValue(params['vx0'])
        self.le_vy0.setValue(params['vy0'])
        self.le_t_start.setValue(params['t_start'])
        self.le_t_end.setValue(params['t_end'])
        self.le_dt_coarse.setValue(params['dt_coarse'])
        self.le_dt_fine.setValue(params['dt_fine'])

    def plot_results(self, results_dict):
        """Строит графики на основе переданных данных."""
        self.plot_data = results_dict
        fig = self.fig_canvas.figure
        fig.clear()

        # Создаем подграфики (3x2)
        axs = fig.subplots(3, 2, squeeze=False) # squeeze=False гарантирует, что axs всегда 2D
        fig.suptitle('Результаты моделирования', fontsize=14)

        if not results_dict:
            fig.canvas.draw()
            return

        # 1. Координата x от времени
        ax = axs[0, 0]
        for label, data in results_dict.items():
            ax.plot(data['t'], data['x'], label=label)
        ax.set_xlabel('Время t')
        ax.set_ylabel('Координата x')
        ax.set_title('x(t)')
        ax.legend(fontsize='small')
        ax.grid(True)

        # 2. Координата y от времени
        ax = axs[0, 1]
        for label, data in results_dict.items():
            ax.plot(data['t'], data['y'], label=label)
        ax.set_xlabel('Время t')
        ax.set_ylabel('Координата y')
        ax.set_title('y(t)')
        ax.legend(fontsize='small')
        ax.grid(True)

        # 3. Проекция скорости vx от времени
        ax = axs[1, 0]
        for label, data in results_dict.items():
            ax.plot(data['t'], data['vx'], label=label)
        ax.set_xlabel('Время t')
        ax.set_ylabel('Скорость vx')
        ax.set_title('vx(t)')
        ax.legend(fontsize='small')
        ax.grid(True)

        # 4. Проекция скорости vy от времени
        ax = axs[1, 1]
        for label, data in results_dict.items():
            ax.plot(data['t'], data['vy'], label=label)
        ax.set_xlabel('Время t')
        ax.set_ylabel('Скорость vy')
        ax.set_title('vy(t)')
        ax.legend(fontsize='small')
        ax.grid(True)

        # 5. Траектория y от x
        ax = axs[2, 0]
        for label, data in results_dict.items():
            ax.plot(data['x'], data['y'], label=label)
        ax.set_xlabel('Координата x')
        ax.set_ylabel('Координата y')
        ax.set_title('Траектория y(x)')
        ax.legend(fontsize='small')
        ax.grid(True)
        # ax.set_aspect('equal', adjustable='box')

        # 6. Траектория x от y
        ax = axs[2, 1]
        for label, data in results_dict.items():
            ax.plot(data['x'], data['y'], label=label)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Траектория x(y)')
        ax.legend(fontsize='small')
        ax.grid(True)
        # ax.set_aspect('equal', adjustable='box')

        fig.tight_layout(pad=2.0)
        fig.canvas.draw()
        
    def enable_animation_controls(self, enable=True):
        """Активирует/деактивирует кнопки анимации."""
        self.btn_animate.setEnabled(enable)
        self.btn_stop_animate.setEnabled(enable)
