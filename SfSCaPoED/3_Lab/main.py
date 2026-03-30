import sys
from PyQt6.QtWidgets import QApplication

from utils.gui.controller import AppController
from utils.gui.resources import STYLESHEET


def main():
    app = QApplication(sys.argv)
    if STYLESHEET:
        app.setStyleSheet(STYLESHEET)

    controller = AppController()
    controller.run()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
