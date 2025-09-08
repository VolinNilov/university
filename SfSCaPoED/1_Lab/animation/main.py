# main.py
import sys
from PyQt6.QtWidgets import QApplication
# Импортируем из пакета gui
from utils.gui.controller import AppController
from utils.gui.resources import STYLESHEET

def main():
    app = QApplication(sys.argv)
    
    # Применяем стили, если есть
    if STYLESHEET:
        app.setStyleSheet(STYLESHEET)
    
    controller = AppController()
    controller.run()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
