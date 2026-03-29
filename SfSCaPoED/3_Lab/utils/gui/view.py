from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QGroupBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QDoubleSpinBox,
    QSpinBox,
)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class InputGroupBox(QGroupBox):
    FIELD_WIDTH = 96

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
        if hasattr(widget, "setMaximumWidth"):
            widget.setMaximumWidth(self.FIELD_WIDTH)
        self.layout().addRow(lbl, widget)


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=7, height=6, dpi=100):
        fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.ax_x = fig.add_subplot(2, 1, 1)
        self.ax_y = fig.add_subplot(2, 1, 2, sharex=self.ax_x)
        fig.subplots_adjust(hspace=0.25, left=0.1, right=0.98, top=0.94, bottom=0.08)
        super().__init__(fig)
        self.setParent(parent)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ЛР3 — EKF")
        self.setGeometry(80, 80, 1280, 820)

        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout(central)

        left_col = QVBoxLayout()
        self._build_form(left_col)
        left_widget = QWidget()
        left_widget.setLayout(left_col)
        left_widget.setMaximumWidth(320)

        self.plot_canvas = PlotCanvas()
        grid.addWidget(left_widget, 0, 0, alignment=Qt.AlignmentFlag.AlignTop)
        grid.addWidget(self.plot_canvas, 0, 1)
        grid.setColumnStretch(1, 1)

    def _build_form(self, col):
        gb_model = InputGroupBox("Модель")
        self.sp_gravity = QDoubleSpinBox()
        self.sp_gravity.setRange(0.01, 50.0)
        self.sp_gravity.setDecimals(4)
        self.sp_friction = QDoubleSpinBox()
        self.sp_friction.setRange(0.0, 10.0)
        self.sp_friction.setDecimals(4)
        self.sp_mass = QDoubleSpinBox()
        self.sp_mass.setRange(0.01, 100.0)
        self.sp_mass.setDecimals(4)
        gb_model.addRow("g", self.sp_gravity)
        gb_model.addRow("b (трение)", self.sp_friction)
        gb_model.addRow("m", self.sp_mass)

        gb_ic = InputGroupBox("Начальные условия")
        self.sp_x0 = QDoubleSpinBox()
        self.sp_x0.setRange(-50.0, 50.0)
        self.sp_x0.setDecimals(6)
        self.sp_y0 = QDoubleSpinBox()
        self.sp_y0.setRange(-50.0, 50.0)
        self.sp_y0.setDecimals(6)
        self.sp_vx0 = QDoubleSpinBox()
        self.sp_vx0.setRange(-50.0, 50.0)
        self.sp_vx0.setDecimals(6)
        self.sp_vy0 = QDoubleSpinBox()
        self.sp_vy0.setRange(-50.0, 50.0)
        self.sp_vy0.setDecimals(6)
        gb_ic.addRow("x₀", self.sp_x0)
        gb_ic.addRow("y₀", self.sp_y0)
        gb_ic.addRow("vₓ₀", self.sp_vx0)
        gb_ic.addRow("vᵧ₀", self.sp_vy0)

        gb_sim = InputGroupBox("Интегрирование")
        self.sp_t_start = QDoubleSpinBox()
        self.sp_t_start.setRange(-100.0, 100.0)
        self.sp_t_start.setDecimals(4)
        self.sp_t_end = QDoubleSpinBox()
        self.sp_t_end.setRange(-100.0, 500.0)
        self.sp_t_end.setDecimals(4)
        self.sp_dt = QDoubleSpinBox()
        self.sp_dt.setRange(1e-5, 1.0)
        self.sp_dt.setDecimals(6)
        self.sp_dt.setSingleStep(0.001)
        gb_sim.addRow("t нач.", self.sp_t_start)
        gb_sim.addRow("t кон.", self.sp_t_end)
        gb_sim.addRow("Δt", self.sp_dt, "Один шаг RK4 и один шаг дискретизации EKF")

        gb_noise = InputGroupBox("Шум по траектории")
        self.sp_noise_sigma = QDoubleSpinBox()
        self.sp_noise_sigma.setRange(0.0, 10.0)
        self.sp_noise_sigma.setDecimals(6)
        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 2_147_483_647)
        gb_noise.addRow("σ", self.sp_noise_sigma)
        gb_noise.addRow("seed", self.sp_seed)

        gb_r = InputGroupBox("R")
        self.sp_sigma_rx = QDoubleSpinBox()
        self.sp_sigma_rx.setRange(1e-8, 10.0)
        self.sp_sigma_rx.setDecimals(6)
        self.sp_sigma_ry = QDoubleSpinBox()
        self.sp_sigma_ry.setRange(1e-8, 10.0)
        self.sp_sigma_ry.setDecimals(6)
        gb_r.addRow("σ_rx", self.sp_sigma_rx, "R = diag(σ_rx², σ_ry²), шаг по x")
        gb_r.addRow("σ_ry", self.sp_sigma_ry, "шаг по y")

        gb_q = InputGroupBox("Q")
        self.sp_q_scale = QDoubleSpinBox()
        self.sp_q_scale.setRange(0.0, 10.0)
        self.sp_q_scale.setDecimals(8)
        gb_q.addRow("q", self.sp_q_scale, "Q = q²I")

        gb_p0 = InputGroupBox("P(0)")
        self.sp_p0_scale = QDoubleSpinBox()
        self.sp_p0_scale.setRange(1e-8, 100.0)
        self.sp_p0_scale.setDecimals(6)
        gb_p0.addRow("p₀", self.sp_p0_scale, "P(0) = p₀²I")

        row_btn = QHBoxLayout()
        self.btn_run = QPushButton("Запустить расчёт")
        row_btn.addWidget(self.btn_run)

        for gb in (gb_model, gb_ic, gb_sim, gb_noise, gb_r, gb_q, gb_p0):
            col.addWidget(gb)
        col.addLayout(row_btn)
        col.addStretch()

    def set_parameters(self, p):
        self.sp_gravity.setValue(p["gravity"])
        self.sp_friction.setValue(p["friction_coeff"])
        self.sp_mass.setValue(p["mass"])
        self.sp_x0.setValue(p["x0"])
        self.sp_y0.setValue(p["y0"])
        self.sp_vx0.setValue(p["vx0"])
        self.sp_vy0.setValue(p["vy0"])
        self.sp_t_start.setValue(p["t_start"])
        self.sp_t_end.setValue(p["t_end"])
        self.sp_dt.setValue(p["dt"])
        self.sp_noise_sigma.setValue(p["noise_sigma"])
        self.sp_seed.setValue(p["random_seed"])
        self.sp_sigma_rx.setValue(p["sigma_r_x"])
        self.sp_sigma_ry.setValue(p["sigma_r_y"])
        self.sp_q_scale.setValue(p["q_scale"])
        self.sp_p0_scale.setValue(p["p0_scale"])

    def get_parameters(self):
        return {
            "gravity": self.sp_gravity.value(),
            "friction_coeff": self.sp_friction.value(),
            "mass": self.sp_mass.value(),
            "x0": self.sp_x0.value(),
            "y0": self.sp_y0.value(),
            "vx0": self.sp_vx0.value(),
            "vy0": self.sp_vy0.value(),
            "t_start": self.sp_t_start.value(),
            "t_end": self.sp_t_end.value(),
            "dt": self.sp_dt.value(),
            "noise_sigma": self.sp_noise_sigma.value(),
            "random_seed": self.sp_seed.value(),
            "sigma_r_x": self.sp_sigma_rx.value(),
            "sigma_r_y": self.sp_sigma_ry.value(),
            "q_scale": self.sp_q_scale.value(),
            "p0_scale": self.sp_p0_scale.value(),
        }

    def plot_results(self, r):
        t = r["t"]
        ax_x = self.plot_canvas.ax_x
        ax_y = self.plot_canvas.ax_y
        ax_x.clear()
        ax_y.clear()

        ax_x.plot(t, r["x_theory"], label="теория", color="#1b5e20", linewidth=1.5)
        ax_x.plot(t, r["x_exp"], label="эксперимент", color="#c62828", alpha=0.85, linewidth=1.0)
        ax_x.plot(t, r["x_filt"], label="EKF", color="#1565c0", linewidth=1.3)
        ax_x.set_ylabel("x(t)")
        ax_x.grid(True, alpha=0.35)
        ax_x.legend(loc="best", fontsize=9)

        ax_y.plot(t, r["y_theory"], label="теория", color="#1b5e20", linewidth=1.5)
        ax_y.plot(t, r["y_exp"], label="эксперимент", color="#c62828", alpha=0.85, linewidth=1.0)
        ax_y.plot(t, r["y_filt"], label="EKF", color="#1565c0", linewidth=1.3)
        ax_y.set_xlabel("t, с")
        ax_y.set_ylabel("y(t)")
        ax_y.grid(True, alpha=0.35)
        ax_y.legend(loc="best", fontsize=9)

        self.plot_canvas.draw()
