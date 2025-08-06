from PyQt5.QtCore import QThread, pyqtSignal
from core.model import auto_train_and_evaluate_models
import os
import sys

class ModelTrainingWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list, dict)
    error_occurred = pyqtSignal(str)  # Signal to notify errors as string messages

    def __init__(self, x, y, prediction_type="classification", train_func=None):
        super().__init__()
        self.x = x
        self.y = y
        self.prediction_type = prediction_type
        self.train_func = train_func
        self.error = None  # Initialize error attribute

    def run(self):
        def progress_callback(percent, name):
            self.progress.emit(percent, f"Training {name}...")

        try:
            # CRITICAL FIX: Set matplotlib backend to non-interactive to prevent window spawning
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Ensure no QApplication instance is created in this thread
            if hasattr(self, '_prevent_qapp_creation'):
                os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            
            # Disable any potential GUI operations during training
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning)
            
            # Import matplotlib after setting backend
            import matplotlib.pyplot as plt
            plt.ioff()  # Turn off interactive mode
            
            if self.train_func is not None:
                results, trained_models = self.train_func(self.x, self.y, self.prediction_type, progress_callback)
            else:
                results, trained_models = auto_train_and_evaluate_models(self.x, self.y, self.prediction_type, progress_callback)

            # Clean up matplotlib
            plt.close('all')
            
            self.finished.emit(results, trained_models)

        except Exception as e:
            self.error = e
            self.error_occurred.emit(str(e))  # Notify UI about the error
        finally:
            # Ensure cleanup
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except:
                pass
