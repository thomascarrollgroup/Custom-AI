from PyQt5.QtCore import QRunnable, QObject, pyqtSignal, QThreadPool
from PyQt5.QtWidgets import QApplication
import pandas as pd
from typing import Callable, Any, Optional, Dict
import traceback
import time
import numpy as np
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


def process_large_dataset_in_chunks(df: pd.DataFrame, chunk_size: int = None, 
                                   operation_func: Callable = None, **kwargs) -> pd.DataFrame:
    """
    Process large datasets in chunks to manage memory.
    
    Args:
        df: DataFrame to process
        chunk_size: Size of each chunk (default from config)
        operation_func: Function to apply to each chunk
        **kwargs: Additional arguments for operation_func
    
    Returns:
        Processed DataFrame or combined results
    """
    if chunk_size is None:
        from core.config import Config
        chunk_size = getattr(Config.ui, 'CHUNK_SIZE', 10000)  # Default to 10k if not configured
    
    if len(df) <= chunk_size:
        if operation_func:
            return operation_func(df, **kwargs)
        return df
    
    logger = get_logger(__name__)
    logger.info(f"Processing large dataset ({len(df)} rows) in chunks of {chunk_size}")
    
    if operation_func is None:
        # If no operation function provided, just return a sample
        logger.warning(f"Dataset sampled to {chunk_size} rows for performance")
        return df.sample(n=min(chunk_size, len(df)), random_state=42)
    
    # Process in chunks
    results = []
    total_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(df), chunk_size):
        chunk_num = i // chunk_size + 1
        logger.info(f"Processing chunk {chunk_num}/{total_chunks}")
        
        chunk = df.iloc[i:i + chunk_size]
        try:
            result = operation_func(chunk, **kwargs)
            if result is not None:
                results.append(result)
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_num}: {e}")
            continue
    
    if results:
        # Combine results if they are DataFrames
        if isinstance(results[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        else:
            return results
    else:
        logger.warning("No results from chunk processing")
        return df.head(0)  # Return empty DataFrame with same structure


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Memory-optimized DataFrame
    """
    logger = get_logger(__name__)
    original_memory = df.memory_usage(deep=True).sum()
    
    # Create a copy to avoid modifying original
    optimized_df = df.copy()
    
    # Optimize numeric columns
    for col in optimized_df.select_dtypes(include=['int64']).columns:
        col_min = optimized_df[col].min()
        col_max = optimized_df[col].max()
        
        if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
            optimized_df[col] = optimized_df[col].astype(np.int8)
        elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
            optimized_df[col] = optimized_df[col].astype(np.int16)
        elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
            optimized_df[col] = optimized_df[col].astype(np.int32)
    
    for col in optimized_df.select_dtypes(include=['float64']).columns:
        optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
    
    # Convert object columns to category where appropriate
    for col in optimized_df.select_dtypes(include=['object']).columns:
        if optimized_df[col].nunique() / len(optimized_df) < 0.5:  # Less than 50% unique values
            optimized_df[col] = optimized_df[col].astype('category')
    
    new_memory = optimized_df.memory_usage(deep=True).sum()
    memory_reduction = (original_memory - new_memory) / original_memory * 100
    
    if memory_reduction > 5:  # Only log if significant reduction
        logger.info(f"Memory usage reduced by {memory_reduction:.1f}% "
                   f"({original_memory / 1024 / 1024:.1f}MB -> {new_memory / 1024 / 1024:.1f}MB)")
    
    return optimized_df