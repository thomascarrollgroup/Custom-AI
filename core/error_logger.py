import os
import traceback
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
from core.config import Config

class ErrorLogger:
    """File-based error logging system to replace Neon DB dependency."""
    
    def __init__(self, log_file_path: str = "error.log", admin_email: str = Config.admin.ADMIN_EMAIL):
        self.log_file_path = Path(log_file_path)
        self.admin_email = admin_email
        self._ensure_log_file_exists()
    
    def _ensure_log_file_exists(self):
        """Ensure the error log file exists and is writable."""
        try:
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_file_path.exists():
                self.log_file_path.touch()
        except Exception as e:
            print(f"Warning: Could not create error log file: {e}")
    
    def log_error(self, user_name: str, exc_type: type, exc_value: Exception, tb) -> Dict[str, Any]:
        """
        Log error to file and return error details for popup.
        
        Returns:
            dict: Error details including formatted message for user display
        """
        try:
            # Extract traceback information
            extracted_tb = traceback.extract_tb(tb)
            if extracted_tb:
                last_frame = extracted_tb[-1]
                file_name = last_frame.filename
                line_number = last_frame.lineno
            else:
                file_name = "Unknown"
                line_number = -1
            
            # Format error details
            error_details = {
                "timestamp": datetime.now().isoformat(),
                "user_name": str(user_name)[:100] if user_name else "Unknown",
                "error_type": exc_type.__name__[:50],
                "error_message": str(exc_value)[:500],
                "file_name": str(file_name)[:200],
                "line_number": line_number,
                "full_traceback": traceback.format_exception(exc_type, exc_value, tb)
            }
            
            # Write to log file
            self._write_to_log_file(error_details)
            
            return error_details
            
        except Exception as e:
            print(f"Failed to log error: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "user_name": str(user_name) if user_name else "Unknown",
                "error_type": "LoggingError",
                "error_message": f"Failed to log original error: {e}",
                "file_name": "error_logger.py",
                "line_number": -1,
                "full_traceback": []
            }
    
    def _write_to_log_file(self, error_details: Dict[str, Any]):
        """Write error details to the log file."""
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                # Write as JSON for structured logging
                json.dump(error_details, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"Failed to write to error log file: {e}")
    
    def get_recent_errors(self, limit: int = 10) -> list:
        """Get recent errors from the log file."""
        try:
            if not self.log_file_path.exists():
                return []
            
            errors = []
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in reversed(lines[-limit:]):
                    try:
                        error = json.loads(line.strip())
                        errors.append(error)
                    except json.JSONDecodeError:
                        continue
            return errors
        except Exception as e:
            print(f"Failed to read error log: {e}")
            return []
    
    def get_log_file_path(self) -> str:
        """Get the path to the error log file."""
        return str(self.log_file_path.absolute())


# Global error logger instance
_error_logger: Optional[ErrorLogger] = None


def get_error_logger(admin_email: str = "admin@company.com") -> ErrorLogger:
    """Get the global error logger instance."""
    global _error_logger
    if _error_logger is None:
        _error_logger = ErrorLogger(admin_email=admin_email)
    return _error_logger


def log_error_to_file(user_name: str, exc_type: type, exc_value: Exception, tb, 
                     admin_email: str = "admin@company.com") -> Dict[str, Any]:
    """
    Log error to file (replacement for log_error_to_neon).
    
    Returns:
        dict: Error details for popup display
    """
    logger = get_error_logger(admin_email)
    return logger.log_error(user_name, exc_type, exc_value, tb)