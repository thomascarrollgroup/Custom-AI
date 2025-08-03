import pandas as pd
from pathlib import Path
from typing import Optional
import chardet

from core.config import Config
from core.errors import DataLoadingError, ValidationError, SecurityError


class DataLoader:
    """Class for loading and validating data files."""

    def load_file(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)
        info = self.get_file_info(path)

        if not path.exists():
            raise DataLoadingError("File not found.")

        if not info['is_supported']:
            raise SecurityError("File type not allowed.")

        if info['file_size_mb'] > Config.security.MAX_FILE_SIZE_MB:
            raise SecurityError("File too large.")

        try:
            if info['file_extension'] in ['.csv']:
                df = pd.read_csv(path, encoding=self.detect_encoding(path))
            elif info['file_extension'] in ['.xlsx', '.xls']:
                df = pd.read_excel(path)
            else:
                raise SecurityError("Unsupported file type.")
        except UnicodeDecodeError:
            raise DataLoadingError("Encoding error while reading file.")
        except MemoryError:
            raise DataLoadingError("File too large to load into memory.")
        except Exception as e:
            raise DataLoadingError("Failed to parse file.", str(e))

        if df.empty:
            raise DataLoadingError("Loaded file is empty.")

        self.validate(df)
        return df

    def validate(self, df: pd.DataFrame) -> None:
        if df.shape[0] > Config.ml.MAX_ROWS:
            raise ValidationError("DataFrame too large.")

        suspicious_names = {'__import__', 'eval', 'exec', 'drop', 'delete'}
        for col in df.columns:
            if any(suspicious in col.lower() for suspicious in suspicious_names):
                raise SecurityError(f"Suspicious column name detected: {col}")

    def get_file_info(self, file_path: str | Path) -> dict:
        path = Path(file_path)
        extension = path.suffix.lower()
        size_mb = path.stat().st_size / (1024 * 1024)

        return {
            "file_extension": extension,
            "is_supported": extension in Config.security.ALLOWED_FILE_EXTENSIONS,
            "file_size_mb": size_mb,
            "estimated_columns": self._estimate_column_count(path, extension)
        }

    def _estimate_column_count(self, path: Path, extension: str) -> Optional[int]:
        try:
            if extension == ".csv":
                with path.open("r", encoding=self.detect_encoding(path)) as f:
                    header = f.readline()
                    return len(header.split(","))
            elif extension in [".xlsx", ".xls"]:
                df = pd.read_excel(path, nrows=1)
                return df.shape[1]
        except Exception:
            return None

    def detect_encoding(self, path: Path) -> str:
        """Detect file encoding for CSV files."""
        with path.open('rb') as f:
            raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'


def load_data_file(file_path: str) -> pd.DataFrame:
    """Convenience function for quick file loading."""
    loader = DataLoader()
    return loader.load_file(file_path)
