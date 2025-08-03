from dataclasses import dataclass, field
import os
from pathlib import Path
import json
from typing import Optional

# Load environment variables from .env file in the project root
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


@dataclass
class MLConfig:
    """Machine learning configuration parameters."""
    MISSING_ROW_THRESHOLD: float = 0.5
    NUMERIC_IMPUTE_VALUE: float = -1
    CATEGORICAL_IMPUTE_VALUE: str = "missing"
    MAX_UNIQUE_CATEGORIES: int = 10
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 3
    MAX_ITERATIONS: int = 1000
    MAX_ROWS: int = 100000  # Maximum allowed rows for performance


@dataclass
class UIConfig:
    """User interface configuration."""
    WINDOW_TITLE: str = "TC AI Prediction Tool"
    DEFAULT_FONT_SIZE: int = 15
    BUTTON_PADDING: str = "16px 24px"
    DEFAULT_WINDOW_WIDTH: int = 1200
    DEFAULT_WINDOW_HEIGHT: int = 800
    CHUNK_SIZE: int = 10000  # For large dataset processing


@dataclass
class SecurityConfig:
    """Security-related configuration."""
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_FILE_EXTENSIONS: tuple = ('.csv', '.xlsx', '.xls')
    CONNECTION_TIMEOUT: int = 5
    MAX_ERROR_MESSAGE_LENGTH: int = 500
    MAX_USERNAME_LENGTH: int = 100


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    NEON_CONN_STR: str = os.getenv("NEON_CONN_STR")
    CONNECTION_TIMEOUT: int = 5
    MAX_RETRY_ATTEMPTS: int = 3

    def is_configured(self) -> bool:
        """Check if database is properly configured."""
        return bool(self.NEON_CONN_STR)


@dataclass
class AppConfig:
    """Main application configuration."""
    ml: MLConfig = field(default_factory=MLConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'AppConfig':
        """Load configuration from JSON file."""
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    data = json.load(f)

                # Create instances with loaded data
                ml_config = MLConfig(**data.get('ml', {}))
                ui_config = UIConfig(**data.get('ui', {}))
                security_config = SecurityConfig(**data.get('security', {}))
                database_config = DatabaseConfig(**data.get('database', {}))

                return cls(ml=ml_config, ui=ui_config, security=security_config, database=database_config)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")
                print("Using default configuration.")

        return cls()

    def save(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        config_data = {
            'ml': self.ml.__dict__,
            'ui': self.ui.__dict__,
            'security': self.security.__dict__,
            'database': self.database.__dict__
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)


# Global configuration instance
Config = AppConfig.load()
