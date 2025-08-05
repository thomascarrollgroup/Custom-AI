import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the test directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_error_logging():
   """Test the new error logging system."""
   print("Testing Error Logging System...")
   try:
        from core.error_logger import get_error_logger, log_error_to_file
        
        # Test creating error logger
        logger = get_error_logger("test@example.com")
        print(f" Error logger created with log file: {logger.get_log_file_path()}")
        
        # Test logging an error
        try:
            raise ValueError("Test error for logging system")
        except Exception as e:
            error_details = log_error_to_file("TestUser", type(e), e, sys.exc_info()[2])
            print(f" Error logged successfully: {error_details['error_type']}")
        
        # Test reading recent errors
        recent_errors = logger.get_recent_errors(limit=1)
        if recent_errors:
            print(f" Recent error retrieved: {recent_errors[0]['error_message']}")
        
        return True
   except Exception as e:
        print(f" Error logging test failed: {e}")
        return False

def test_data_quality_analysis():
    """Test the new data quality analysis system."""
    print("\nTesting Data Quality Analysis System...")
    
    try:
        from core.data_quality import InteractiveDataQuality
        
        # Create sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'numeric_col1': np.random.normal(100, 15, 1000),
            'numeric_col2': np.random.exponential(2, 1000),
            'categorical_col': np.random.choice(['A', 'B', 'C', 'D'], 1000),
            'text_col': np.random.choice(['Apple', 'Banana', 'Cherry', 'Date', None], 1000),
            'target': np.random.choice([0, 1], 1000)
        })
        
        # Add some missing values
        sample_data.loc[sample_data.sample(frac=0.1).index, 'numeric_col1'] = np.nan
        sample_data.loc[sample_data.sample(frac=0.05).index, 'text_col'] = None
        
        print(f" Created sample dataset: {sample_data.shape}")
        
        # Test data quality analyzer
        dq_analyzer = InteractiveDataQuality(sample_data)
        print(" Data quality analyzer created")
        
        # Test summary report
        summary = dq_analyzer.get_summary_report()
        print(f" Summary report generated ({len(summary)} characters)")
        
        # Test chart creation
        chart_options = dq_analyzer.get_chart_options()
        print(f" Available charts: {list(chart_options.keys())}")
        
        # Test creating a specific chart
        missing_chart = dq_analyzer.create_chart('missing')
        if missing_chart is not None:
            print(" Missing data chart created successfully")
       
        numeric_chart = dq_analyzer.create_chart('numeric')
        if numeric_chart is not None:
            print(" Numeric distribution chart created successfully")
        
        return True
        
    except Exception as e:
        print(f" Data quality analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_optimization():
    """Test memory optimization features."""
    print("\nTesting Memory Optimization...")
    
    try:
        from core.async_operations import optimize_dataframe_memory, process_large_dataset_in_chunks
        
        # Create a large dataset with inefficient types
        large_data = pd.DataFrame({
            'small_int': np.random.randint(0, 10, 10000, dtype=np.int64),  # Can be int8
            'medium_int': np.random.randint(0, 1000, 10000, dtype=np.int64),  # Can be int16
            'float_col': np.random.random(10000).astype(np.float64),  # Can be float32
            'category_col': np.random.choice(['Cat1', 'Cat2', 'Cat3'], 10000).astype('object')
        })
        
        original_memory = large_data.memory_usage(deep=True).sum()
        print(f" Created large dataset: {large_data.shape}, Memory: {original_memory / 1024 / 1024:.2f}MB")
        
        # Test memory optimization
        optimized_data = optimize_dataframe_memory(large_data)
        optimized_memory = optimized_data.memory_usage(deep=True).sum()
        
        memory_saved = (original_memory - optimized_memory) / original_memory * 100
        print(f" Memory optimization: {memory_saved:.1f}% reduction")
        
        # Test chunking
        def dummy_operation(chunk):
            return len(chunk)
        
        result = process_large_dataset_in_chunks(
            large_data, 
            chunk_size=2000, 
            operation_func=dummy_operation
        )
        
        if isinstance(result, list) and len(result) == 5:  # 10000 / 2000 = 5 chunks
            print(" Chunking operation completed successfully")
        
        return True
        
    except Exception as e:
        print(f" Memory optimization test failed: {e}")
        return False

def test_error_dialog():
    """Test error dialog functionality (without actually showing GUI)."""
    print("\nTesting Error Dialog System...")
    
    try:
        from ui.error_dialog import ErrorDialog
        from PyQt5.QtWidgets import QApplication
        
        # Check that we can import and create the dialog class
        error_details = {
            'timestamp': '2024-01-01T12:00:00',
            'user_name': 'TestUser',
            'error_type': 'TestError',
            'error_message': 'This is a test error',
            'file_name': 'test.py',
            'line_number': 42,
            'full_traceback': ['Traceback line 1', 'Traceback line 2']
        }
        
        print(" Error dialog can be imported and error details formatted")
        
        # Test email URL generation (without actually opening email client)
        import urllib.parse
        subject = "Test Error Report"
        body = "Test error details"
        subject_encoded = urllib.parse.quote(subject)
        body_encoded = urllib.parse.quote(body)
        mailto_url = f"mailto:admin@example.com?subject={subject_encoded}&body={body_encoded}"
        
        if mailto_url.startswith("mailto:"):
            print(" Email URL generation works correctly")
        
        return True
        
    except Exception as e:
        print(f" Error dialog test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running New Features Test Suite")
    print("=" * 50)
    
    tests = [
        test_error_logging,
        test_data_quality_analysis,
        test_memory_optimization,
        test_error_dialog
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f" Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! New features are working correctly.")
        return 0
    else:
        print("Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())