from PyQt5.QtCore import QThread, pyqtSignal
from core.model import auto_train_and_evaluate_models

class ModelTrainingWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list, dict)
    def __init__(self, x, y, prediction_type="classification", train_func=None):
        super().__init__()
        self.x = x
        self.y = y
        self.prediction_type = prediction_type
        self.train_func = train_func

    def run(self):
        def progress_callback(percent, name):
            self.progress.emit(percent, f"Training {name}...")
        if self.train_func is not None:
            results, trained_models = self.train_func(self.x, self.y, self.prediction_type, progress_callback)
        else:
            from core.model import auto_train_and_evaluate_models
            results, trained_models = auto_train_and_evaluate_models(self.x, self.y, self.prediction_type, progress_callback)
        self.finished.emit(results, trained_models)