from enum import Enum
from typing import Optional
import logging

class ErrorType(Enum):
    """Enumeration of application error types."""
    DATA_LOADING = "data_loading"
    PREPROCESSING = "preprocessing"
    MODEL_TRAINING = "model_training"
    PREDICTION = "prediction"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    VALIDATION = "validation"

class AppError(Exception):
    """Base application error with categorization."""
    
    def __init__(self, error_type: ErrorType, message: str, details: Optional[str] = None):
        self.error_type = error_type
        self.message = message
        self.details = details
        super().__init__(message)

class DataLoadingError(AppError):
    """Errors related to loading data files."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(ErrorType.DATA_LOADING, message, details)

class PreprocessingError(AppError):
    """Errors related to data preprocessing."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(ErrorType.PREPROCESSING, message, details)

class ModelTrainingError(AppError):
    """Errors related to model training."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(ErrorType.MODEL_TRAINING, message, details)

class PredictionError(AppError):
    """Errors related to making predictions."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(ErrorType.PREDICTION, message, details)

class SecurityError(AppError):
    """Errors related to security violations."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(ErrorType.SECURITY, message, details)

class ValidationError(AppError):
    """Errors related to data validation."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(ErrorType.VALIDATION, message, details)

def handle_error(error: Exception, context: str, logger: Optional[logging.Logger] = None) -> None:
    """Centralized error handling with proper logging."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if isinstance(error, AppError):
        logger.error(f"{error.error_type.value} in {context}: {error.message}")
        if error.details:
            logger.debug(f"Error details: {error.details}")
    else:
        logger.error(f"Unexpected error in {context}: {str(error)}")
        logger.debug(f"Error type: {type(error).__name__}")
