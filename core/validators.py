import pandas as pd
import os
from pathlib import Path
from typing import Union
from core.config import Config
from core.errors import ValidationError, SecurityError


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and sanitize input DataFrame."""
    if df is None:
        raise ValidationError("DataFrame cannot be None")
    
    if df.empty:
        raise ValidationError("DataFrame cannot be empty")
    
    if df.shape[0] > Config.security.MAX_ROWS:
        raise ValidationError(
            f"DataFrame too large: {df.shape[0]} rows (max: {Config.security.MAX_ROWS})",
            f"Consider reducing your dataset size or processing in chunks"
        )
    
    # Check for suspicious column names that might indicate malicious data
    suspicious_patterns = ['__', 'eval', 'exec', 'import', 'os.', 'sys.']
    for col in df.columns:
        col_str = str(col).lower()
        if any(pattern in col_str for pattern in suspicious_patterns):
            raise SecurityError(
                f"Suspicious column name detected: {col}",
                "Column names should not contain executable code patterns"
            )
    
    return df


def validate_file_path(file_path: Union[str, Path]) -> Path:
    """Validate file path for security and existence."""
    if not file_path:
        raise ValidationError("File path cannot be empty")
    
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        raise ValidationError(f"File does not exist: {file_path}")
    
    # Check file extension
    if path.suffix.lower() not in Config.security.ALLOWED_FILE_EXTENSIONS:
        raise SecurityError(
            f"File type not allowed: {path.suffix}",
            f"Allowed types: {', '.join(Config.security.ALLOWED_FILE_EXTENSIONS)}"
        )
    
    # Check file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > Config.security.MAX_FILE_SIZE_MB:
        raise SecurityError(
            f"File too large: {file_size_mb:.1f}MB (max: {Config.security.MAX_FILE_SIZE_MB}MB)"
        )
    
    # Security check: ensure file is not in sensitive directories
    resolved_path = path.resolve()
    sensitive_dirs = ['/etc', '/sys', '/proc', '/dev']
    if any(str(resolved_path).startswith(sensitive_dir) for sensitive_dir in sensitive_dirs):
        raise SecurityError(f"Access to file in sensitive directory denied: {resolved_path}")
    
    return path


def validate_target_column(df: pd.DataFrame, target_col: str) -> str:
    """Validate target column selection."""
    if not target_col:
        raise ValidationError("Target column cannot be empty")
    
    if target_col not in df.columns:
        raise ValidationError(
            f"Target column '{target_col}' not found in dataset",
            f"Available columns: {', '.join(df.columns.tolist())}"
        )
    
    # Check if target column has sufficient non-null values
    non_null_ratio = df[target_col].count() / len(df)
    if non_null_ratio < 0.1:  # Less than 10% non-null values
        raise ValidationError(
            f"Target column '{target_col}' has too many missing values ({non_null_ratio:.1%} non-null)"
        )
    
    return target_col


def validate_features(df: pd.DataFrame, features: list) -> list:
    """Validate feature selection."""
    if not features:
        raise ValidationError("At least one feature must be selected")
    
    if len(features) > len(df.columns) - 1:  # -1 for target column
        raise ValidationError("Too many features selected")
    
    missing_features = [feat for feat in features if feat not in df.columns]
    if missing_features:
        raise ValidationError(
            f"Features not found in dataset: {', '.join(missing_features)}"
        )
    
    return features


def sanitize_username(username: str) -> str:
    """Sanitize username for database storage."""
    if not username:
        return "Unknown"
    
    # Remove potential SQL injection characters and limit length
    sanitized = ''.join(c for c in username if c.isalnum() or c in '_-.')
    return sanitized[:Config.security.MAX_USERNAME_LENGTH]


def validate_model_name(model_name: str) -> str:
    """Validate model name for file system safety."""
    if not model_name:
        raise ValidationError("Model name cannot be empty")
    
    # Remove unsafe characters for file names
    safe_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.')
    if not all(c in safe_chars for c in model_name):
        raise ValidationError(
            "Model name contains invalid characters",
            "Use only letters, numbers, underscore, hyphen, and period"
        )
    
    if len(model_name) > 50:
        raise ValidationError("Model name too long (max 50 characters)")
    
    return model_name