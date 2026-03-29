import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGroupBox, QFormLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QSplitter, QSizePolicy,
    QDoubleSpinBox, QSpinBox, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer, QRect, QSize
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPalette, QBrush
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

from utils.model import ConstraintMotionModel

class InputGroupBox(QGroupBox):
    """Группа для ввода параметров"""

    # Компактная ширина полей ввода (чтобы не тянулись на всю колонку)
    FIELD_WIDTH = 88

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        self.setLayout(layout)

    def addRow(self, label_text, widget, tooltip=None):
        """Добавляет строку: подпись + виджет. tooltip задаёт подсказку для подписи и поля."""
        lbl = QLabel(label_text)
        if tooltip:
            lbl.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            # Длительность показа подсказки (мс) — на виджете, не на QApplication
            for w in (lbl, widget):
                if hasattr(w, "setToolTipDuration"):
                    w.setToolTipDuration(5000)
        if hasattr(widget, "setMaximumWidth"):
            widget.setMaximumWidth(self.FIELD_WIDTH)
        self.layout().addRow(lbl, widget)

class PlotCanvas(FigureCanvas):
    """Холст для matplotlib"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

class AnimationWidget(QWidget):
    """Виджет для анимации"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px;")
        # Жёстко задаём светлый фон, чтобы не зависеть от темы ОС
        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window, QColor(248, 249, 250))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

        self.x_data = np.array([])
        self.y_data = np.array([])
        self.t_data = np.array([])
        self.current_index = 0
        self.bbox = ConstraintMotionModel.constraint_bbox()
        self.curve_x, self.curve_y = ConstraintMotionModel.constraint_curve_xy()

        # Таймер для анимации
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.animation_speed = 50 # мс

    def set_data(self, data_dict):
        """Устанавливает данные для анимации (ожидает ключи x, y, t; опционально _bbox)."""

        self.x_data = np.array(data_dict.get("x", []))
        self.y_data = np.array(data_dict.get("y", []))
        self.t_data = np.array(data_dict.get("t", []))
        self.current_index = 0
        bb = data_dict.get("_bbox")
        self.bbox = tuple(bb) if bb is not None else ConstraintMotionModel.constraint_bbox()
        self.curve_x, self.curve_y = ConstraintMotionModel.constraint_curve_xy()
        self.update()

    def start_animation(self):
        if len(self.x_data) > 1:
            self.current_index = 0
            self.timer.start(self.animation_speed)

    def stop_animation(self):
        self.timer.stop()

    def update_animation(self):
        if self.current_index < len(self.x_data) - 1:
            self.current_index += 1
            self.update()
        else:
            self.timer.stop() # Остановить в конце

    def _clamp_px(self, v, lo=-2**31 + 1, hi=2**31 - 1):
        """Ограничивает координату пикселя в диапазон int32, чтобы QPainter не падал с OverflowError."""
        return max(lo, min(hi, int(np.round(v))))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        """Сохраняем квадрат: высота = ширина."""
        return max(self.minimumHeight(), w)

    def sizeHint(self):
        s = 360
        return QSize(s, s)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        painter.fillRect(self.rect(), QColor(248, 249, 250))

        # Область для текста — с отступами, чтобы надписи помещались в виджет
        text_rect = self.rect().adjusted(20, 20, -20, -20)
        wrap_center = Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap

        if len(self.x_data) == 0 or len(self.y_data) == 0:
            painter.setPen(QColor(108, 117, 125))
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(text_rect, wrap_center, "Запустите расчёт и анимацию")
            return

        # Если в данных есть NaN/Inf (неустойчивое решение)
        used_x = self.x_data[: self.current_index + 1] if self.current_index < len(self.x_data) else self.x_data
        used_y = self.y_data[: self.current_index + 1] if self.current_index < len(self.y_data) else self.y_data
        if not np.all(np.isfinite(used_x)) or not np.all(np.isfinite(used_y)):
            painter.setPen(QColor(108, 117, 125))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(
                text_rect, wrap_center,
                "Решение неустойчиво (NaN/Inf). Уменьшите шаг dt или измените параметры."
            )
            return

        width = self.width()
        height = self.height()
        margin = 24
        xmin, xmax, ymin, ymax = self.bbox
        w_world = max(xmax - xmin, 1e-6)
        h_world = max(ymax - ymin, 1e-6)
        scale = min((width - 2 * margin) / w_world, (height - 2 * margin) / h_world)
        cx_world = 0.5 * (xmin + xmax)
        cy_world = 0.5 * (ymin + ymax)
        center_x = width // 2
        center_y = height // 2
        clamp = self._clamp_px

        def world_to_px(wx, wy):
            sx = center_x + (wx - cx_world) * scale
            sy = center_y - (wy - cy_world) * scale
            return clamp(sx), clamp(sy)

        # Контур связи f=0
        painter.setPen(QPen(QColor(73, 80, 87), 2, Qt.PenStyle.SolidLine))
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        if len(self.curve_x) > 1:
            for i in range(1, len(self.curve_x)):
                x1, y1 = world_to_px(self.curve_x[i - 1], self.curve_y[i - 1])
                x2, y2 = world_to_px(self.curve_x[i], self.curve_y[i])
                painter.drawLine(x1, y1, x2, y2)

        # Траектория
        if self.current_index > 0:
            painter.setPen(QPen(QColor(30, 64, 175), 2, Qt.PenStyle.SolidLine))
            for i in range(1, self.current_index + 1):
                x1, y1 = world_to_px(self.x_data[i - 1], self.y_data[i - 1])
                x2, y2 = world_to_px(self.x_data[i], self.y_data[i])
                painter.drawLine(x1, y1, x2, y2)

        if 0 <= self.current_index < len(self.x_data):
            obj_x, obj_y = world_to_px(self.x_data[self.current_index], self.y_data[self.current_index])

            painter.setPen(QPen(QColor(220, 53, 69), 2, Qt.PenStyle.SolidLine))
            painter.setBrush(QBrush(QColor(220, 53, 69)))
            painter.drawEllipse(clamp(obj_x - 6), clamp(obj_y - 6), 12, 12)

            time_str = f"t={self.t_data[self.current_index]:.3f} s" if self.current_index < len(self.t_data) else f"i={self.current_index}"
            font = QFont("Segoe UI", 9)
            font.setBold(True)
            painter.setFont(font)
            fm = painter.fontMetrics()
            text_rect = fm.boundingRect(time_str)
            pad_x, pad_y = 6, 2
            box = QRect(
                clamp(obj_x + 10),
                clamp(obj_y - text_rect.height() // 2 - pad_y),
                max(1, text_rect.width() + 2 * pad_x),
                max(1, text_rect.height() + 2 * pad_y)
            )
            painter.setBrush(QColor(255, 255, 255))
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            painter.drawRoundedRect(box, 4, 4)
            painter.setPen(QColor(33, 37, 41))
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, time_str)

class MainWindow(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа 1 - Моделирование движения")
        self.setGeometry(100, 100, 1400, 900)

        # Центральный виджет — сетка 2x2: [Настройки|График], [Анимация|График]
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        grid = QGridLayout(central_widget)

        # Левая колонка — шире, чтобы настройки и окно анимации (квадрат) помещались
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(380)
        left_panel.setMaximumWidth(440)

        # Группа параметров модели (с подсказками к параметрам)
        self.model_group = InputGroupBox("Параметры модели")
        self.le_gravity = QDoubleSpinBox()
        self.le_gravity.setRange(0.1, 50.0)
        self.le_gravity.setSingleStep(0.1)
        self.le_gravity.setValue(9.81)
        self.model_group.addRow("Ускорение (g):", self.le_gravity, "Ускорение свободного падения (м/с²)")

        self.le_friction = QDoubleSpinBox()
        self.le_friction.setRange(0.0, 10.0)
        self.le_friction.setSingleStep(0.01)
        self.le_friction.setValue(0.2)
        self.model_group.addRow("Трение (b):", self.le_friction, "Коэффициент трения (демпфирования)")

        self.le_mass = QDoubleSpinBox()
        self.le_mass.setRange(0.1, 100.0)
        self.le_mass.setSingleStep(0.1)
        self.le_mass.setValue(1.0)
        self.model_group.addRow("Масса (m):", self.le_mass, "Масса шарика (кг)")
        left_layout.addWidget(self.model_group)

        # Группа начальных условий
        self.ic_group = InputGroupBox("Начальные условия")
        self.le_x0 = QDoubleSpinBox()
        self.le_x0.setRange(-100.0, 100.0)
        self.le_x0.setSingleStep(0.1)
        self.le_x0.setDecimals(4)
        self.le_x0.setValue(0.0)
        self.ic_group.addRow("x0:", self.le_x0, "Начальная координата x (на связи f=0)")

        self.le_y0 = QDoubleSpinBox()
        self.le_y0.setRange(-100.0, 100.0)
        self.le_y0.setSingleStep(0.1)
        self.le_y0.setDecimals(4)
        self.le_y0.setValue(4.0)
        self.ic_group.addRow("y0:", self.le_y0, "Начальная координата y (на связи f=0)")

        self.le_vx0 = QDoubleSpinBox()
        self.le_vx0.setRange(-100.0, 100.0)
        self.le_vx0.setSingleStep(0.1)
        self.le_vx0.setValue(0.0)
        self.ic_group.addRow("Vx0:", self.le_vx0, "Начальная скорость по оси x (м/с)")

        self.le_vy0 = QDoubleSpinBox()
        self.le_vy0.setRange(-100.0, 100.0)
        self.le_vy0.setSingleStep(0.1)
        self.le_vy0.setValue(0.0)
        self.ic_group.addRow("Vy0:", self.le_vy0, "Начальная скорость по оси y (м/с)")
        left_layout.addWidget(self.ic_group)

        # Группа параметров интегрирования
        self.int_group = InputGroupBox("Параметры интегрирования")
        self.le_t_start = QDoubleSpinBox()
        self.le_t_start.setRange(0.0, 1000.0)
        self.le_t_start.setSingleStep(0.1)
        self.le_t_start.setValue(0.0)
        self.int_group.addRow("t_start:", self.le_t_start, "Начало интервала интегрирования (с)")

        self.le_t_end = QDoubleSpinBox()
        self.le_t_end.setRange(0.1, 1000.0)
        self.le_t_end.setSingleStep(0.1)
        self.le_t_end.setValue(5.0)
        self.int_group.addRow("t_end:", self.le_t_end, "Конец интервала интегрирования (с)")

        self.le_dt_coarse = QDoubleSpinBox()
        self.le_dt_coarse.setRange(0.0001, 1.0)
        self.le_dt_coarse.setSingleStep(0.001)
        self.le_dt_coarse.setDecimals(4)
        self.le_dt_coarse.setValue(0.05)
        self.int_group.addRow("dt (грубый):", self.le_dt_coarse, "Шаг для методов Euler, Heun, RK4 (с)")

        self.le_dt_fine = QDoubleSpinBox()
        self.le_dt_fine.setRange(0.0001, 1.0)
        self.le_dt_fine.setSingleStep(0.0001)
        self.le_dt_fine.setDecimals(5)
        self.le_dt_fine.setValue(0.005)
        self.int_group.addRow("dt (мелкий):", self.le_dt_fine, "Мелкий шаг для эталонного решения RK4 (с)")
        left_layout.addWidget(self.int_group)

        # Кнопки управления
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

        # Левая колонка, строка 1: окно анимации (квадрат 360×360, шире чем раньше)
        self.animation_widget = AnimationWidget()
        self.animation_widget.setMinimumSize(360, 360)
        self.animation_widget.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )

        # Правая колонка (на обе строки): график результатов
        self.plot_tabs = QWidget()
        plot_layout = QVBoxLayout(self.plot_tabs)
        self.fig_canvas = FigureCanvas(plt.Figure(figsize=(8, 6)))
        plot_layout.addWidget(self.fig_canvas)

        # Сетка 2x2: (0,0)=настройки, (1,0)=анимация, (0,1)-(1,1)=график на всю высоту
        grid.addWidget(left_panel, 0, 0)
        grid.addWidget(self.animation_widget, 1, 0)
        grid.addWidget(self.plot_tabs, 0, 1, 2, 1)  # rowSpan=2 — график на всю 2-ю колонку
        grid.setColumnMinimumWidth(0, 380)  # левая колонка шире, чтобы анимация была квадратом
        grid.setColumnStretch(1, 1)   # колонка с графиком тянется
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        # Хранилище для данных графиков
        self.plot_data = {}

    def get_parameters(self):
        """Собирает параметры из полей ввода"""

        return {
            "gravity": self.le_gravity.value(),
            "friction_coeff": self.le_friction.value(),
            "mass": self.le_mass.value(),
            "x0": self.le_x0.value(),
            "y0": self.le_y0.value(),
            "vx0": self.le_vx0.value(),
            "vy0": self.le_vy0.value(),
            "t_start": self.le_t_start.value(),
            "t_end": self.le_t_end.value(),
            "dt_coarse": self.le_dt_coarse.value(),
            "dt_fine": self.le_dt_fine.value(),
        }

    def set_parameters(self, params):
        """Устанавливает параметры в поля ввода"""

        self.le_gravity.setValue(params["gravity"])
        self.le_friction.setValue(params["friction_coeff"])
        self.le_mass.setValue(params["mass"])
        self.le_x0.setValue(params["x0"])
        self.le_y0.setValue(params["y0"])
        self.le_vx0.setValue(params["vx0"])
        self.le_vy0.setValue(params['vy0'])
        self.le_t_start.setValue(params['t_start'])
        self.le_t_end.setValue(params['t_end'])
        self.le_dt_coarse.setValue(params['dt_coarse'])
        self.le_dt_fine.setValue(params['dt_fine'])

    def plot_results(self, results_dict):
        """Строит графики на основе переданных данных"""

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
        """Активирует/деактивирует кнопки анимации"""
        
        self.btn_animate.setEnabled(enable)
        self.btn_stop_animate.setEnabled(enable)
