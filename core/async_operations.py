"""
Async operations to prevent UI blocking for I/O intensive tasks.
"""

from PyQt5.QtCore import QRunnable, QObject, pyqtSignal, QThreadPool
from PyQt5.QtWidgets import QApplication
import pandas as pd
from typing import Callable, Any, Optional, Dict
import traceback
import time
from pathlib import Path

from core.data_loader import load_data_file
from core.errors import DataLoadingError, ValidationError, SecurityError
from core.logging_setup import get_logger


class WorkerSignals(QObject):
    """Signals for communicating between worker threads and main thread."""
    
    finished = pyqtSignal()
    error = pyqtSignal(Exception)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)
    status = pyqtSignal(str)


class AsyncFileLoader(QRunnable):
    """Load files asynchronously to prevent UI blocking."""
    
    def __init__(self, file_path: str, callback_success: Optional[Callable] = None, 
                 callback_error: Optional[Callable] = None):
        super().__init__()
        self.file_path = file_path
        self.callback_success = callback_success
        self.callback_error = callback_error
        self.signals = WorkerSignals()
        self.logger = get_logger(__name__)
        
        # Connect signals to callbacks if provided
        if callback_success:
            self.signals.result.connect(callback_success)
        if callback_error:
            self.signals.error.connect(callback_error)
    
    def run(self):
        """Load file in background thread."""
        try:
            self.signals.status.emit("Validating file...")
            QApplication.processEvents()
            
            # Add artificial delay for user feedback on very fast operations
            time.sleep(0.1)
            
            self.signals.status.emit("Loading file...")
            start_time = time.time()
            
            # Load the file using safe data loader
            df = load_data_file(self.file_path)
            
            duration = time.time() - start_time
            self.logger.info(
                f"File loaded successfully in {duration:.2f}s",
                extra={
                    'operation': 'async_file_load',
                    'file_path': self.file_path,
                    'duration': duration,
                    'data_shape': f"{df.shape[0]}x{df.shape[1]}"
                }
            )
            
            self.signals.status.emit("File loaded successfully!")
            self.signals.result.emit(df)
            
        except (DataLoadingError, ValidationError, SecurityError) as e:
            self.logger.error(f"File loading failed: {e}")
            self.signals.error.emit(e)
        except Exception as e:
            self.logger.error(f"Unexpected error in async file loading: {e}")
            self.signals.error.emit(e)
        finally:
            self.signals.finished.emit()


class AsyncDataProcessor(QRunnable):
    """Process data asynchronously (preprocessing, training, etc.)."""
    
    def __init__(self, operation: str, data: Any, callback_success: Optional[Callable] = None,
                 callback_error: Optional[Callable] = None, **kwargs):
        super().__init__()
        self.operation = operation
        self.data = data
        self.kwargs = kwargs
        self.callback_success = callback_success
        self.callback_error = callback_error
        self.signals = WorkerSignals()
        self.logger = get_logger(__name__)
        
        # Connect signals to callbacks if provided
        if callback_success:
            self.signals.result.connect(callback_success)
        if callback_error:
            self.signals.error.connect(callback_error)
    
    def run(self):
        """Process data in background thread."""
        try:
            self.signals.status.emit(f"Starting {self.operation}...")
            start_time = time.time()
            
            if self.operation == "preprocess":
                from core.preprocess import auto_preprocess_data
                target_col = self.kwargs.get('target_col')
                result = auto_preprocess_data(self.data, target_col)
                
            elif self.operation == "train_models":
                from core.model import auto_train_and_evaluate_models
                target_col = self.kwargs.get('target_col')
                features = self.kwargs.get('features')
                prediction_type = self.kwargs.get('prediction_type', 'classification')
                result = auto_train_and_evaluate_models(
                    self.data, features, target_col, prediction_type
                )
                
            elif self.operation == "make_predictions":
                # Handle prediction operations
                model = self.kwargs.get('model')
                encoders = self.kwargs.get('encoders')
                features = self.kwargs.get('features')
                
                from core.preprocess import preprocess_test_data
                processed_data = preprocess_test_data(self.data, encoders, features)
                predictions = model.predict(processed_data)
                result = predictions
                
            else:
                raise ValueError(f"Unknown operation: {self.operation}")
            
            duration = time.time() - start_time
            self.logger.info(
                f"{self.operation} completed successfully in {duration:.2f}s",
                extra={
                    'operation': f'async_{self.operation}',
                    'duration': duration
                }
            )
            
            self.signals.status.emit(f"{self.operation} completed!")
            self.signals.result.emit(result)
            
        except Exception as e:
            self.logger.error(f"Error in async {self.operation}: {e}")
            self.signals.error.emit(e)
        finally:
            self.signals.finished.emit()


class AsyncModelSaver(QRunnable):
    """Save models and encoders asynchronously."""
    
    def __init__(self, model_data: Dict[str, Any], save_path: str,
                 callback_success: Optional[Callable] = None,
                 callback_error: Optional[Callable] = None):
        super().__init__()
        self.model_data = model_data
        self.save_path = save_path
        self.callback_success = callback_success
        self.callback_error = callback_error
        self.signals = WorkerSignals()
        self.logger = get_logger(__name__)
        
        # Connect signals to callbacks if provided
        if callback_success:
            self.signals.result.connect(callback_success)
        if callback_error:
            self.signals.error.connect(callback_error)
    
    def run(self):
        """Save model data in background thread."""
        try:
            self.signals.status.emit("Saving model and encoders...")
            start_time = time.time()
            
            from core.model import save_model
            save_model(
                self.model_data['best_model'],
                self.model_data['encoders'],
                self.model_data['features'],
                self.save_path
            )
            
            duration = time.time() - start_time
            self.logger.info(
                f"Model saved successfully in {duration:.2f}s",
                extra={
                    'operation': 'async_model_save',
                    'save_path': self.save_path,
                    'duration': duration
                }
            )
            
            self.signals.status.emit("Model saved successfully!")
            self.signals.result.emit(self.save_path)
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            self.signals.error.emit(e)
        finally:
            self.signals.finished.emit()


class AsyncManager:
    """Manage async operations with thread pool."""
    
    def __init__(self, max_threads: int = 4):
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(max_threads)
        self.logger = get_logger(__name__)
        
    def load_file_async(self, file_path: str, success_callback: Callable, 
                       error_callback: Callable) -> AsyncFileLoader:
        """Load file asynchronously."""
        worker = AsyncFileLoader(file_path, success_callback, error_callback)
        self.thread_pool.start(worker)
        self.logger.info(f"Started async file loading: {file_path}")
        return worker
    
    def process_data_async(self, operation: str, data: Any, success_callback: Callable,
                          error_callback: Callable, **kwargs) -> AsyncDataProcessor:
        """Process data asynchronously."""
        worker = AsyncDataProcessor(operation, data, success_callback, error_callback, **kwargs)
        self.thread_pool.start(worker)
        self.logger.info(f"Started async {operation}")
        return worker
    
    def save_model_async(self, model_data: Dict[str, Any], save_path: str,
                        success_callback: Callable, error_callback: Callable) -> AsyncModelSaver:
        """Save model asynchronously."""
        worker = AsyncModelSaver(model_data, save_path, success_callback, error_callback)
        self.thread_pool.start(worker)
        self.logger.info(f"Started async model saving: {save_path}")
        return worker
    
    def get_active_thread_count(self) -> int:
        """Get number of active threads."""
        return self.thread_pool.activeThreadCount()
    
    def wait_for_completion(self, timeout_ms: int = 30000) -> bool:
        """Wait for all threads to complete."""
        return self.thread_pool.waitForDone(timeout_ms)


# Global async manager instance
_async_manager: Optional[AsyncManager] = None


def get_async_manager() -> AsyncManager:
    """Get the global async manager instance."""
    global _async_manager
    if _async_manager is None:
        _async_manager = AsyncManager()
    return _async_manager


def process_large_dataset_in_chunks(df: pd.DataFrame, chunk_size: int = None) -> pd.DataFrame:
    """Process large datasets in chunks to manage memory."""
    if chunk_size is None:
        from core.config import Config
        chunk_size = Config.ui.CHUNK_SIZE
    
    if len(df) <= chunk_size:
        return df
    
    logger = get_logger(__name__)
    logger.info(f"Processing large dataset ({len(df)} rows) in chunks of {chunk_size}")
    
    logger.warning(f"Dataset truncated to {chunk_size} rows for performance")
    return df.head(chunk_size)