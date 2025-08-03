"""
Comprehensive logging setup with structured logging and metrics collection.
"""
import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from core.config import Config


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'user_name'):
            log_entry['user_name'] = record.user_name
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration
        if hasattr(record, 'data_shape'):
            log_entry['data_shape'] = record.data_shape
            
        return json.dumps(log_entry)


class MetricsCollector:
    """Collect application metrics for monitoring."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.metrics")
    
    def record_training_time(self, model_name: str, duration: float) -> None:
        """Record model training duration."""
        metric_key = f"training_time_{model_name}"
        self.metrics[metric_key] = duration
        self.logger.info(
            "Model training completed", 
            extra={
                'operation': 'model_training',
                'model_name': model_name,
                'duration': duration
            }
        )
    
    def record_data_size(self, rows: int, columns: int) -> None:
        """Record dataset dimensions."""
        self.metrics.update({"data_rows": rows, "data_columns": columns})
        self.logger.info(
            "Dataset loaded",
            extra={
                'operation': 'data_loading',
                'data_shape': f"{rows}x{columns}"
            }
        )
    
    def record_prediction_batch(self, batch_size: int, duration: float) -> None:
        """Record prediction batch processing."""
        self.logger.info(
            "Predictions generated",
            extra={
                'operation': 'prediction',
                'batch_size': batch_size,
                'duration': duration
            }
        )
    
    def record_error(self, error_type: str, context: str, user_name: Optional[str] = None) -> None:
        """Record error occurrence."""
        self.logger.error(
            f"Error in {context}",
            extra={
                'operation': 'error',
                'error_type': error_type,
                'context': context,
                'user_name': user_name or 'unknown'
            }
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        return self.metrics.copy()


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = True
) -> MetricsCollector:
    """
    Configure comprehensive logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: logs/ in project root)
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_json: Use JSON format for structured logging
        
    Returns:
        MetricsCollector instance for application metrics
    """
    
    # Create log directory
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Choose formatter based on JSON setting
    if enable_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Use simple format for console even if JSON is enabled for files
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handlers
    if enable_file:
        # General application log
        app_log_file = log_dir / "app.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        app_handler.setLevel(logging.INFO)
        app_handler.setFormatter(formatter)
        root_logger.addHandler(app_handler)
        
        # Error log
        error_log_file = log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        
        # Security log
        security_log_file = log_dir / "security.log"
        security_handler = logging.handlers.RotatingFileHandler(
            security_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10  # Keep more security logs
        )
        security_handler.setLevel(logging.WARNING)
        security_handler.setFormatter(formatter)
        
        # Add security handler only to security-related loggers
        security_logger = logging.getLogger("security")
        security_logger.addHandler(security_handler)
        security_logger.setLevel(logging.WARNING)
    
    # Configure specific loggers
    loggers_config = {
        "core.data_loader": logging.INFO,
        "core.preprocess": logging.INFO,
        "core.model": logging.INFO,
        "core.safe_analysis": logging.INFO,
        "ui.app": logging.INFO,
        "security": logging.WARNING
    }
    
    for logger_name, level in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
    
    # Log startup
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging system initialized",
        extra={
            'operation': 'system_startup',
            'log_level': log_level,
            'log_dir': str(log_dir),
            'console_enabled': enable_console,
            'file_enabled': enable_file,
            'json_enabled': enable_json
        }
    )
    
    return MetricsCollector()


def get_logger(name: str, user_name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with optional user context.
    
    Args:
        name: Logger name (usually __name__)
        user_name: Optional user name for context
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if user_name:
        # Create adapter to add user context to all log messages
        class UserContextAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                extra = kwargs.get('extra', {})
                extra['user_name'] = self.extra['user_name']
                kwargs['extra'] = extra
                return msg, kwargs
        
        return UserContextAdapter(logger, {'user_name': user_name})
    
    return logger


def log_security_event(event_type: str, description: str, user_name: Optional[str] = None, **kwargs):
    """Log security-related events."""
    security_logger = logging.getLogger("security")
    security_logger.warning(
        f"Security event: {event_type} - {description}",
        extra={
            'operation': 'security_event',
            'event_type': event_type,
            'user_name': user_name or 'unknown',
            **kwargs
        }
    )


# Global metrics collector instance
metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global metrics_collector
    if metrics_collector is None:
        metrics_collector = MetricsCollector()
    return metrics_collector