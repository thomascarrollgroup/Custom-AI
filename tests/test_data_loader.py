import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from core.data_loader import DataLoader, load_data_file
from core.errors import DataLoadingError, ValidationError, SecurityError


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'category_col': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_csv_loading_success(self):
        """Test successful CSV loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            df = self.loader.load_file(temp_path)
            assert len(df) == 5
            assert list(df.columns) == ['numeric_col', 'category_col', 'target']
            assert df['numeric_col'].dtype in ['int64', 'int32']
        finally:
            os.unlink(temp_path)
    
    def test_excel_loading_success(self):
        """Test successful Excel loading."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.xlsx', delete=False) as f:
            self.test_data.to_excel(f.name, index=False)
            temp_path = f.name
        
        try:
            df = self.loader.load_file(temp_path)
            assert len(df) == 5
            assert list(df.columns) == ['numeric_col', 'category_col', 'target']
        finally:
            os.unlink(temp_path)
    
    def test_file_not_found(self):
        """Test handling of non-existent files."""
        with pytest.raises(DataLoadingError, match="File not found"):
            self.loader.load_file("nonexistent_file.csv")
    
    def test_unsupported_file_type(self):
        """Test handling of unsupported file types."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            with pytest.raises(SecurityError, match="File type not allowed"):
                self.loader.load_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_empty_csv_file(self):
        """Test handling of empty CSV files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name
        
        try:
            with pytest.raises(DataLoadingError, match="empty"):
                self.loader.load_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_malformed_csv_file(self):
        """Test handling of malformed CSV files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n1,2,3,4\n5,6")  # Inconsistent columns
            temp_path = f.name
        
        try:
            with pytest.raises(DataLoadingError, match="parse"):
                self.loader.load_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_file_size_limit(self):
        """Test file size validation."""
        # Mock a very large file
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 100 * 1024 * 1024  # 100MB
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write("col1\n1\n")
                temp_path = f.name
            
            try:
                with pytest.raises(SecurityError, match="File too large"):
                    self.loader.load_file(temp_path)
            finally:
                os.unlink(temp_path)
    
    def test_get_file_info(self):
        """Test file information retrieval."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            info = self.loader.get_file_info(temp_path)
            assert info['file_extension'] == '.csv'
            assert info['is_supported'] is True
            assert 'file_size_mb' in info
            assert info['estimated_columns'] == 3
        finally:
            os.unlink(temp_path)
    
    def test_convenience_function(self):
        """Test convenience function."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            df = load_data_file(temp_path)
            assert len(df) == 5
        finally:
            os.unlink(temp_path)


class TestDataValidation:
    """Test data validation during loading."""
    
    def test_suspicious_column_names(self):
        """Test detection of suspicious column names."""
        suspicious_data = pd.DataFrame({
            '__import__': [1, 2, 3],
            'eval_col': [4, 5, 6],
            'normal_col': [7, 8, 9]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            suspicious_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            with pytest.raises(SecurityError, match="Suspicious column name"):
                load_data_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_very_large_dataset(self):
        """Test handling of very large datasets."""
        # Create a large dataset that exceeds the limit
        large_data = pd.DataFrame({
            'col1': range(200000),  # Assuming MAX_ROWS is 100000
            'col2': range(200000)
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            with pytest.raises(ValidationError, match="DataFrame too large"):
                load_data_file(temp_path)
        finally:
            os.unlink(temp_path)


class TestErrorHandling:
    """Test error handling in data loading."""
    
    def test_permission_error(self):
        """Test handling of permission errors."""
        with patch('core.data_loader.load_data_file', side_effect=PermissionError("Access denied")):
            loader = DataLoader()
            with pytest.raises(SecurityError, match="Permission denied"):
                loader.load_file("some_file.csv")
    
    def test_memory_error_csv(self):
        """Test handling of memory errors during CSV loading."""
        with patch('pandas.read_csv', side_effect=MemoryError("Not enough memory")):
            loader = DataLoader()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write("col1\n1\n")
                temp_path = f.name
            
            try:
                with pytest.raises(DataLoadingError, match="File too large to load into memory"):
                    loader.load_file(temp_path)
            finally:
                os.unlink(temp_path)
    
    def test_encoding_error(self):
        """Test handling of encoding errors."""
        # Create a file with non-UTF-8 content
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as f:
            # Write some non-UTF-8 bytes
            f.write(b'col1\n\xff\xfe\n')
            temp_path = f.name
        
        try:
            with pytest.raises(DataLoadingError, match="Encoding error"):
                loader = DataLoader()
                loader.load_file(temp_path)
        finally:
            os.unlink(temp_path)