from PyQt6.QtCore import QObject

from utils.gui.model import AppModel
from utils.gui.view import MainWindow
from utils.gui.resources import DEFAULT_PARAMS


class AppController(QObject):

    def __init__(self):
        super().__init__()
        self.view = MainWindow()
        self.model = AppModel()

        self.view.btn_run.clicked.connect(self.on_run_clicked)
        self.view.btn_animate.clicked.connect(self.on_animate_clicked)
        self.view.btn_stop_animate.clicked.connect(self.on_stop_animate_clicked)

        self.model.calculationFinished.connect(self.on_calculation_finished)
        self.model.calculationStarted.connect(self.on_calculation_started)
        self.model.calculationError.connect(self.on_calculation_error)
        self.model.animationDataReady.connect(self.on_animation_data_ready)

        self.view.set_parameters(DEFAULT_PARAMS)
        self.animation_data = None

    def run(self):
        self.view.show()

    def on_run_clicked(self):
        params = self.view.get_parameters()
        self.model.set_parameters(params)
        self.model.run_calculations()

    def on_animate_clicked(self):
        if self.animation_data:
            self.view.animation_widget.set_data(self.animation_data)
            self.view.animation_widget.start_animation()
            self.view.btn_animate.setEnabled(False)
            self.view.btn_stop_animate.setEnabled(True)

    def on_stop_animate_clicked(self):
        self.view.animation_widget.stop_animation()
        self.view.btn_animate.setEnabled(True)
        self.view.btn_stop_animate.setEnabled(False)

    def on_calculation_started(self):
        self.view.btn_run.setText("Расчет...")
        self.view.btn_run.setEnabled(False)
        self.view.enable_animation_controls(False)

    def on_calculation_finished(self, results):
        self.view.btn_run.setText("Запустить расчеты")
        self.view.btn_run.setEnabled(True)
        self.view.plot_results(results)
        self.view.enable_animation_controls(True)
        print("[Controller] Расчеты завершены, графики обновлены.")

    def on_calculation_error(self, error_msg):
        self.view.btn_run.setText("Запустить расчеты")
        self.view.btn_run.setEnabled(True)
        print(f"[Controller] Ошибка расчета: {error_msg}")

    def on_animation_data_ready(self, data):
        self.animation_data = data
        print("[Controller] Данные для анимации получены.")
