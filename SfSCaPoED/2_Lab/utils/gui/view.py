import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGroupBox, QFormLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QSplitter, QSizePolicy,
    QDoubleSpinBox, QSpinBox, QGridLayout, QComboBox,
)
from PyQt6.QtCore import Qt, QTimer, QRect, QSize
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPalette, QBrush
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

from utils.model import ConstraintMotionModel


class InputGroupBox(QGroupBox):
    FIELD_WIDTH = 88

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        self.setLayout(layout)

    def addRow(self, label_text, widget, tooltip=None):
        lbl = QLabel(label_text)
        if tooltip:
            lbl.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            for w in (lbl, widget):
                if hasattr(w, "setToolTipDuration"):
                    w.setToolTipDuration(5000)
        if hasattr(widget, "setMaximumWidth"):
            widget.setMaximumWidth(self.FIELD_WIDTH)
        self.layout().addRow(lbl, widget)

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

class AnimationWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px;")
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

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.animation_speed = 50

    def set_data(self, data_dict):
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
            self.timer.stop()

    def _clamp_px(self, v, lo=-2**31 + 1, hi=2**31 - 1):
        return max(lo, min(hi, int(np.round(v))))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return max(self.minimumHeight(), w)

    def sizeHint(self):
        s = 360
        return QSize(s, s)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        painter.fillRect(self.rect(), QColor(248, 249, 250))

        text_rect = self.rect().adjusted(20, 20, -20, -20)
        wrap_center = Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap

        if len(self.x_data) == 0 or len(self.y_data) == 0:
            painter.setPen(QColor(108, 117, 125))
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(text_rect, wrap_center, "Запустите расчёт и анимацию")
            return

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

        painter.setPen(QPen(QColor(73, 80, 87), 2, Qt.PenStyle.SolidLine))
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        if len(self.curve_x) > 1:
            for i in range(1, len(self.curve_x)):
                x1, y1 = world_to_px(self.curve_x[i - 1], self.curve_y[i - 1])
                x2, y2 = world_to_px(self.curve_x[i], self.curve_y[i])
                painter.drawLine(x1, y1, x2, y2)

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

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа 2 — шум и доверительные интервалы")
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        grid = QGridLayout(central_widget)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(380)
        left_panel.setMaximumWidth(440)

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

        self.le_dt_fine = QDoubleSpinBox()
        self.le_dt_fine.setRange(0.0001, 1.0)
        self.le_dt_fine.setSingleStep(0.0001)
        self.le_dt_fine.setDecimals(5)
        self.le_dt_fine.setValue(0.005)
        self.int_group.addRow("dt:", self.le_dt_fine, "Шаг RK4 для теоретической траектории (с)")
        left_layout.addWidget(self.int_group)

        self.exp_group = InputGroupBox("Эксперимент (шум)")
        self.combo_n = QComboBox()
        self.combo_n.addItem("5 траекторий", 5)
        self.combo_n.addItem("25 траекторий", 25)
        self.exp_group.addRow("N:", self.combo_n, "Число экспериментальных реализаций")

        self.le_noise_sigma = QDoubleSpinBox()
        self.le_noise_sigma.setRange(0.0, 10.0)
        self.le_noise_sigma.setSingleStep(0.005)
        self.le_noise_sigma.setDecimals(4)
        self.le_noise_sigma.setValue(0.02)
        self.exp_group.addRow("σ шума:", self.le_noise_sigma, "СКО приращений накопленного шума (норм. распр.)")

        self.le_seed = QSpinBox()
        self.le_seed.setRange(0, 2_147_483_647)
        self.le_seed.setValue(42)
        self.exp_group.addRow("seed:", self.le_seed, "База для генератора (воспроизводимость)")
        left_layout.addWidget(self.exp_group)

        self.btn_run = QPushButton("Запустить расчеты")
        self.btn_run.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(self.btn_run)
        
        self.btn_animate = QPushButton("Запустить анимацию")
        self.btn_animate.setEnabled(False)
        left_layout.addWidget(self.btn_animate)
        
        self.btn_stop_animate = QPushButton("Остановить анимацию")
        self.btn_stop_animate.setEnabled(False)
        left_layout.addWidget(self.btn_stop_animate)

        left_layout.addStretch()

        self.animation_widget = AnimationWidget()
        self.animation_widget.setMinimumSize(360, 360)
        self.animation_widget.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )

        self.plot_tabs = QWidget()
        plot_layout = QVBoxLayout(self.plot_tabs)
        self.fig_canvas = FigureCanvas(plt.Figure(figsize=(8, 6)))
        plot_layout.addWidget(self.fig_canvas)

        grid.addWidget(left_panel, 0, 0)
        grid.addWidget(self.animation_widget, 1, 0)
        grid.addWidget(self.plot_tabs, 0, 1, 2, 1)
        grid.setColumnMinimumWidth(0, 380)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        self.plot_data = {}

    def get_parameters(self):
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
            "dt_fine": self.le_dt_fine.value(),
            "n_runs": int(self.combo_n.currentData()),
            "noise_sigma": self.le_noise_sigma.value(),
            "random_seed": self.le_seed.value(),
        }

    def set_parameters(self, params):
        self.le_gravity.setValue(params["gravity"])
        self.le_friction.setValue(params["friction_coeff"])
        self.le_mass.setValue(params["mass"])
        self.le_x0.setValue(params["x0"])
        self.le_y0.setValue(params["y0"])
        self.le_vx0.setValue(params["vx0"])
        self.le_vy0.setValue(params["vy0"])
        self.le_t_start.setValue(params["t_start"])
        self.le_t_end.setValue(params["t_end"])
        self.le_dt_fine.setValue(params["dt_fine"])
        n = int(params.get("n_runs", 5))
        for i in range(self.combo_n.count()):
            if self.combo_n.itemData(i) == n:
                self.combo_n.setCurrentIndex(i)
                break
        self.le_noise_sigma.setValue(params.get("noise_sigma", 0.02))
        self.le_seed.setValue(int(params.get("random_seed", 42)))

    def plot_results(self, results_dict):
        self.plot_data = results_dict
        fig = self.fig_canvas.figure
        fig.clear()

        if not results_dict or not results_dict.get("lab2"):
            fig.canvas.draw()
            return

        t = np.asarray(results_dict["t"], dtype=float)
        x_th = np.asarray(results_dict["x_theory"], dtype=float)
        y_th = np.asarray(results_dict["y_theory"], dtype=float)
        x_exp = np.asarray(results_dict["x_exp"], dtype=float)
        y_exp = np.asarray(results_dict["y_exp"], dtype=float)
        x_mean = np.asarray(results_dict["x_mean"], dtype=float)
        y_mean = np.asarray(results_dict["y_mean"], dtype=float)
        x_lo = np.asarray(results_dict["x_ci_lo"], dtype=float)
        x_hi = np.asarray(results_dict["x_ci_hi"], dtype=float)
        y_lo = np.asarray(results_dict["y_ci_lo"], dtype=float)
        y_hi = np.asarray(results_dict["y_ci_hi"], dtype=float)
        N = int(results_dict["N"])
        sigma = float(results_dict["noise_sigma"])
        alpha = float(results_dict["alpha"])
        dt_f = float(results_dict.get("dt_fine", 0.005))

        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.28)
        ax_xy = fig.add_subplot(gs[0, :])
        n_traj, _ = x_exp.shape
        for k in range(n_traj):
            ax_xy.plot(x_exp[k], y_exp[k], color="C0", alpha=0.35, linewidth=0.9)
        ax_xy.plot(x_th, y_th, "k--", linewidth=1.5, label="Теория (RK4)")
        ax_xy.plot(x_mean, y_mean, "r-", linewidth=2.2, label="Среднее по экспериментам")
        cx, cy = ConstraintMotionModel.constraint_curve_xy()
        ax_xy.plot(cx, cy, color="gray", linewidth=1.0, alpha=0.5, linestyle=":", label="Связь f=0")
        ax_xy.set_xlabel("x")
        ax_xy.set_ylabel("y")
        ax_xy.set_title("Плоскость xy")
        ax_xy.legend(loc="best", fontsize="small")
        ax_xy.grid(True)

        ax_x = fig.add_subplot(gs[1, 0])
        ax_x.fill_between(t, x_lo, x_hi, color="C1", alpha=0.35, label=f"ДИ {(1-alpha)*100:.0f}% (Стьюдент)")
        ax_x.plot(t, x_mean, color="C1", linewidth=2.0, label="Среднее по эксп. x(t)")
        ax_x.plot(t, x_th, "k--", linewidth=1.0, alpha=0.65, label="Теория x(t)")
        ax_x.set_xlabel("t")
        ax_x.set_ylabel("x")
        ax_x.set_title("x(t) и доверительный интервал")
        ax_x.legend(loc="best", fontsize="small")
        ax_x.grid(True)

        ax_y = fig.add_subplot(gs[1, 1])
        ax_y.fill_between(t, y_lo, y_hi, color="C2", alpha=0.35, label=f"ДИ {(1-alpha)*100:.0f}% (Стьюдент)")
        ax_y.plot(t, y_mean, color="C2", linewidth=2.0, label="Среднее по эксп. y(t)")
        ax_y.plot(t, y_th, "k--", linewidth=1.0, alpha=0.65, label="Теория y(t)")
        ax_y.set_xlabel("t")
        ax_y.set_ylabel("y")
        ax_y.set_title("y(t) и доверительный интервал")
        ax_y.legend(loc="best", fontsize="small")
        ax_y.grid(True)

        pct = (1.0 - alpha) * 100.0
        fig.suptitle(
            f"ЛР2: N={N}, σ={sigma:g}, dt={dt_f:g} с, теория RK4, ДИ {pct:.0f}%",
            fontsize=13,
            y=1.02,
        )
        fig.subplots_adjust(top=0.88)
        fig.canvas.draw()

    def enable_animation_controls(self, enable=True):
        self.btn_animate.setEnabled(enable)
        self.btn_stop_animate.setEnabled(enable)
