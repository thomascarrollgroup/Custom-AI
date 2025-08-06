#!/usr/bin/env python3
"""
Debug script to help identify column mapping issues between training and test data.
"""

import pandas as pd
import pickle
import os
from core.preprocess import find_best_column_match, get_required_columns_for_test_data, validate_test_data_columns

def debug_column_mapping(encoder_file_path: str, test_data_path: str):
    """
    Debug column mapping between training encoders and test data.
    
    Args:
        encoder_file_path: Path to the encoder pickle file
        test_data_path: Path to the test data file
    """
    print("=== Column Mapping Debug Tool ===\n")
    
    # Load encoder data
    print(f"Loading encoder file: {encoder_file_path}")
    try:
        with open(encoder_file_path, 'rb') as f:
            enc_data = pickle.load(f)
        
        encoders = enc_data['encoders']
        features = enc_data['features']
        feature_to_base = enc_data.get('feature_to_base', {})
        
        print(f" Encoder file loaded successfully")
        print(f"  - Number of features: {len(features)}")
        print(f"  - Number of encoders: {len(encoders)}")
        print(f"  - Feature to base mapping: {len(feature_to_base)} entries")
        
    except Exception as e:
        print(f" Failed to load encoder file: {e}")
        return
    
    # Load test data
    print(f"\nLoading test data: {test_data_path}")
    try:
        if test_data_path.endswith('.csv'):
            test_df = pd.read_csv(test_data_path)
        elif test_data_path.endswith(('.xlsx', '.xls')):
            test_df = pd.read_excel(test_data_path)
        else:
            print(f" Unsupported file format: {test_data_path}")
            return
        
        print(f" Test data loaded successfully")
        print(f"  - Shape: {test_df.shape}")
        print(f"  - Columns: {list(test_df.columns)}")
        
    except Exception as e:
        print(f" Failed to load test data: {e}")
        return
    
    # Get required columns
    print(f"\n=== Required Columns Analysis ===")
    required_cols = get_required_columns_for_test_data(encoders, feature_to_base)
    print(f"Required columns from training: {required_cols}")
    
    # Validate column mapping
    print(f"\n=== Column Validation ===")
    is_valid, missing_cols, mapped_cols = validate_test_data_columns(test_df, encoders, feature_to_base)
    
    if is_valid:
        print(" All required columns found!")
    else:
        print(" Missing required columns:")
        for col in missing_cols:
            print(f"  - {col}")
    
    if mapped_cols:
        print(f"\nColumn mappings found:")
        for orig, mapped in mapped_cols:
            print(f"  - '{orig}' -> '{mapped}'")
    
    # Detailed analysis
    print(f"\n=== Detailed Column Analysis ===")
    available_cols = list(test_df.columns)
    
    for req_col in required_cols:
        print(f"\nAnalyzing required column: '{req_col}'")
        
        # Check exact match
        if req_col in available_cols:
            print(f"   Exact match found")
            continue
        
        # Check case-insensitive match
        case_matches = [col for col in available_cols if col.lower() == req_col.lower()]
        if case_matches:
            print(f"   Case-insensitive match found: {case_matches}")
            continue
        
        # Try fuzzy matching
        best_match = find_best_column_match(req_col, available_cols, threshold=0.6)
        if best_match:
            similarity = calculate_similarity(req_col, best_match)
            print(f"   Fuzzy match found: '{best_match}' (similarity: {similarity:.2f})")
        else:
            print(f"   No match found")
            
            # Show similar columns
            similarities = []
            for col in available_cols:
                sim = calculate_similarity(req_col, col)
                if sim > 0.3:  # Show columns with >30% similarity
                    similarities.append((col, sim))
            
            if similarities:
                similarities.sort(key=lambda x: x[1], reverse=True)
                print(f"    Similar columns:")
                for col, sim in similarities[:3]:
                    print(f"      - '{col}' (similarity: {sim:.2f})")
    
    # Show one-hot encoded features
    print(f"\n=== One-Hot Encoded Features ===")
    ohe_features = [feat for feat in features if '_' in feat and feat.split('_')[0] in required_cols]
    if ohe_features:
        print(f"One-hot encoded features that will be created:")
        for feat in ohe_features:
            print(f"  - {feat}")
    else:
        print("No one-hot encoded features found")


def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings."""
    from difflib import SequenceMatcher
    import re
    
    # Normalize strings
    norm1 = re.sub(r'[_\s]+', '', str1.lower())
    norm2 = re.sub(r'[_\s]+', '', str2.lower())
    
    return SequenceMatcher(None, norm1, norm2).ratio()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python debug_columns.py <encoder_file.pkl> <test_data.csv>")
        print("\nExample:")
        print("  python debug_columns.py encoders/model_encoders.pkl test_data.csv")
        sys.exit(1)
    
    encoder_file = sys.argv[1]
    test_data_file = sys.argv[2]
    
    if not os.path.exists(encoder_file):
        print(f"Error: Encoder file not found: {encoder_file}")
        sys.exit(1)
    
    if not os.path.exists(test_data_file):
        print(f"Error: Test data file not found: {test_data_file}")
        sys.exit(1)
    
    debug_column_mapping(encoder_file, test_data_file) 