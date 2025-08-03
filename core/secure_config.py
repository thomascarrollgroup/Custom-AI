import os
class ConfigurationError(Exception):
    """Raised when required environment variables are missing."""

class SecureConfig:
    def __init__(self):
        self._validate_environment()

    def _validate_environment(self):
        """Ensure critical environment variables are set."""
        required_vars = ['NEON_CONN_STR']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ConfigurationError(f"Missing environment variables: {missing}")

    @property
    def NEON_CONN_STR(self) -> str:
        return os.getenv("NEON_CONN_STR")
