from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import os
@dataclass
class MLConfig:
    """Machine learning configuration parameters."""
    MISSING_ROW_THRESHOLD: float = 0.5
    NUMERIC_IMPUTE_VALUE: float = -1
    CATEGORICAL_IMPUTE_VALUE: str = "missing"
    MAX_UNIQUE_CATEGORIES: int = 10
    TEST_SIZE: float = 0.5
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
class AdminConfig:
    """Admin configuration."""
    ADMIN_EMAIL: str = "devaang.misra@thomas-carroll.co.uk"


@dataclass
class AppConfig:
    """Main application configuration."""
    ml: MLConfig = field(default_factory=MLConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    admin: AdminConfig = field(default_factory=AdminConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'AppConfig':
        """Load configuration from a JSON file if provided."""
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    data = json.load(f)

                return cls(
                    ml=MLConfig(**data.get('ml', {})),
                    ui=UIConfig(**data.get('ui', {})),
                    admin=AdminConfig(**data.get('admin', {}))
                )
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")
                print("Using default configuration.")
        
        return cls()

    def save(self, config_path: Path) -> None:
        """Save configuration to a JSON file."""
        config_data = {
            'ml': self.ml.__dict__,
            'ui': self.ui.__dict__,
            'admin': self.admin.__dict__
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)


# Global configuration instance
Config = AppConfig.load()
