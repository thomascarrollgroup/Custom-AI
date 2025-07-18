import sys
from PyQt5.QtWidgets import QApplication
from ui.app import BespokePredictionApp
from core.logging import global_exception_hook

if __name__ == "__main__":
    sys.excepthook = global_exception_hook
    app = QApplication(sys.argv)
    window = BespokePredictionApp()
    window.show()
    sys.exit(app.exec_())