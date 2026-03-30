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

        self.model.calculationFinished.connect(self.on_calculation_finished)
        self.model.calculationStarted.connect(self.on_calculation_started)
        self.model.calculationError.connect(self.on_calculation_error)

        self.view.set_parameters(DEFAULT_PARAMS)

    def run(self):
        self.view.show()

    def on_run_clicked(self):
        params = self.view.get_parameters()
        self.model.set_parameters(params)
        self.model.run_calculations()

    def on_calculation_started(self):
        self.view.btn_run.setText("Расчёт...")
        self.view.btn_run.setEnabled(False)

    def on_calculation_finished(self, results):
        self.view.btn_run.setText("Запустить расчёт")
        self.view.btn_run.setEnabled(True)
        self.view.plot_results(results)

    def on_calculation_error(self, error_msg):
        self.view.btn_run.setText("Запустить расчёт")
        self.view.btn_run.setEnabled(True)
        print(error_msg)
