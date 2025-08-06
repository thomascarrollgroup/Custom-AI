import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, List, Any, Optional
from core.config import Config
from difflib import SequenceMatcher

def clean_numeric(val: Any) -> Optional[float]:
    """Extract numeric value from a string"""
    if pd.isnull(val):
        return np.nan
    val = str(val)
    match = re.findall(r'-?\d+\.?\d*', val.replace(' ', ''))
    return float(match[0]) if match else np.nan


def safe_label_encode(encoder: LabelEncoder, data: pd.Series, default_value=None) -> np.ndarray:
    """
    Encoding data with LabelEncoder without mutating its internal state.
    Unknown values are mapped to a fallback label (first class by default).
    """
    known_classes = set(encoder.classes_)
    fallback = default_value if default_value is not None else encoder.classes_[0]
    safe_data = data.astype(str).apply(lambda x: x if x in known_classes else fallback)
    return encoder.transform(safe_data)


def find_best_column_match(target_col: str, available_cols: List[str], threshold: float = 0.8) -> Optional[str]:
    """
    Find the best matching column name using fuzzy string matching.
    
    Args:
        target_col: The target column name to find
        available_cols: List of available column names
        threshold: Minimum similarity score (0-1) to consider a match
    
    Returns:
        The best matching column name or None if no match above threshold
    """
    if target_col in available_cols:
        return target_col
    
    # Try exact match first
    for col in available_cols:
        if col.lower() == target_col.lower():
            return col
    
    # Try fuzzy matching
    best_match = None
    best_score = 0
    
    for col in available_cols:
        # Normalize column names for comparison
        norm_target = re.sub(r'[_\s]+', '', target_col.lower())
        norm_col = re.sub(r'[_\s]+', '', col.lower())
        
        # Calculate similarity
        similarity = SequenceMatcher(None, norm_target, norm_col).ratio()
        
        if similarity > best_score and similarity >= threshold:
            best_score = similarity
            best_match = col
    
    return best_match


def get_required_columns_for_test_data(encoders: Dict[str, Any], feature_to_base: Dict[str, str]) -> List[str]:
    """
    Get the list of required columns for test data based on the training encoders.
    
    Args:
        encoders: Dictionary of encoders from training
        feature_to_base: Mapping of features to base columns
    
    Returns:
        List of required column names for test data
    """
    required_cols = set()
    
    for base_col in set(feature_to_base.values()):
        if base_col in encoders:
            required_cols.add(base_col)
    
    return sorted(list(required_cols))


def validate_test_data_columns(test_df: pd.DataFrame, encoders: Dict[str, Any], feature_to_base: Dict[str, str]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate that test data has the required columns.
    
    Args:
        test_df: Test data DataFrame
        encoders: Dictionary of encoders from training
        feature_to_base: Mapping of features to base columns
    
    Returns:
        Tuple of (is_valid, missing_cols, mapped_cols)
    """
    required_cols = get_required_columns_for_test_data(encoders, feature_to_base)
    available_cols = list(test_df.columns)
    
    missing_cols = []
    mapped_cols = []
    
    for req_col in required_cols:
        actual_col = find_best_column_match(req_col, available_cols)
        if actual_col is None:
            missing_cols.append(req_col)
        elif actual_col != req_col:
            mapped_cols.append((req_col, actual_col))
    
    return len(missing_cols) == 0, missing_cols, mapped_cols


def auto_preprocess_data(
    df: pd.DataFrame,
    target_col: str
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str], Dict[str, str]]:  # Added Dict[str, str] for map
    """
    Auto-preprocess a DataFrame by:

    1. Attempting to convert object columns to numeric if they look like numbers
    2. Encoding categorical columns with LabelEncoder or OneHotEncoder
    3. Imputing missing values with mean or most frequent value
    4. Scaling numeric columns with StandardScaler

    Returns a tuple of:

    1. Preprocessed DataFrame
    2. A dictionary of encoders used for each column
    3. A list of columns that were processed (i.e. not the target column)
    4. A dictionary mapping each feature to its base column (for one-hot encoded columns)
    
    Memory optimization: Uses in-place operations where possible to reduce memory usage.
    """
    # Create a copy only once at the beginning
    df = df.copy()
    encoders: Dict[str, Any] = {}
    processed_cols: List[str] = []
    feature_to_base: Dict[str, str] = {}

    # Attempt to convert object columns to numeric if they look like numbers
    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            cleaned = df[col].map(clean_numeric)
            if cleaned.notnull().sum() > Config.ml.MISSING_ROW_THRESHOLD * len(df):
                df[col] = cleaned
        except Exception:
            pass

    for col in df.columns:
        if col == target_col:
            if df[col].dtype == object or pd.api.types.is_categorical_dtype(df[col]):
                le = LabelEncoder()
                df[col] = df[col].astype(str).fillna(Config.ml.CATEGORICAL_IMPUTE_VALUE)
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            # Use in-place operations to reduce memory usage
            imputer = SimpleImputer(strategy='mean')
            df.loc[:, col] = imputer.fit_transform(df[[col]]).ravel()
            encoders[col + "_imputer"] = imputer

            scaler = StandardScaler()
            df.loc[:, col] = scaler.fit_transform(df[[col]]).ravel()
            encoders[col] = scaler
            processed_cols.append(col)
            feature_to_base[col] = col

        else:
            # Use in-place operations to reduce memory usage
            imputer = SimpleImputer(strategy='most_frequent')
            df.loc[:, col] = imputer.fit_transform(df[[col]]).ravel()
            encoders[col + "_imputer"] = imputer

            nunique = df[col].nunique(dropna=True)
            if nunique <= Config.ml.MAX_UNIQUE_CATEGORIES:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                arr = ohe.fit_transform(df[[col]])
                ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                df = df.drop(columns=[col])
                df[ohe_cols] = arr
                encoders[col] = ohe
                processed_cols.extend(ohe_cols)
                # Map each one-hot col to base col
                for ohe_col in ohe_cols:
                    feature_to_base[ohe_col] = col
            else:
                le = LabelEncoder()
                # Use in-place operations to reduce memory usage
                df.loc[:, col] = df[col].astype(str)
                df.loc[:, col] = le.fit_transform(df[col])
                encoders[col] = le
                processed_cols.append(col)
                feature_to_base[col] = col

    df = df.loc[:, ~df.columns.duplicated()]
    return df, encoders, processed_cols, feature_to_base


def preprocess_test_data(
    test_df: pd.DataFrame,
    encoders: Dict[str, Any],
    features: List[str],
    feature_to_base: Dict[str, str],
    target_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Preprocess test data to exactly match training features,
    including one-hot encoded columns that may not appear in test set.
    """
    df = test_df.copy()
    processed = pd.DataFrame(index=df.index)
    available_cols = list(df.columns)
    
    # Log available columns for debugging
    print(f"Available columns in test data: {available_cols}")
    print(f"Expected base columns from training: {list(set(feature_to_base.values()))}")

    for feat in features:
        base_col = feature_to_base.get(feat, feat)
        encoder = encoders.get(base_col)
        imputer = encoders.get(base_col + "_imputer")

        # Try to find the best matching column in test data
        actual_col = find_best_column_match(base_col, available_cols)
        
        if actual_col is None:
            print(f"Warning: Could not find column '{base_col}' in test data. Using default values.")
            # Column missing entirely in test data, fallback to default values
            if isinstance(encoder, OneHotEncoder):
                processed[feat] = 0
            elif isinstance(encoder, (StandardScaler, LabelEncoder)):
                processed[feat] = Config.ml.NUMERIC_IMPUTE_VALUE
            else:
                processed[feat] = Config.ml.NUMERIC_IMPUTE_VALUE
            continue
        elif actual_col != base_col:
            print(f"Info: Mapped '{base_col}' to '{actual_col}' in test data")

        col_data = df[actual_col].copy()

        # Impute if needed
        if imputer:
            try:
                # Convert to numpy array and reshape for imputer
                if hasattr(col_data, 'values'):
                    col_data_array = col_data.values.reshape(-1, 1)
                else:
                    col_data_array = np.array(col_data).reshape(-1, 1)
                col_data = imputer.transform(col_data_array).ravel()
            except Exception as e:
                print(f"Warning: Imputation failed for '{base_col}': {e}. Using default values.")
                col_data = pd.Series(col_data).fillna(Config.ml.NUMERIC_IMPUTE_VALUE)

        # Apply encoder
        if encoder:
            if isinstance(encoder, StandardScaler):
                try:
                    # Convert to numpy array and reshape for scaler
                    if hasattr(col_data, 'values'):
                        col_data_array = col_data.values.reshape(-1, 1)
                    else:
                        col_data_array = np.array(col_data).reshape(-1, 1)
                    col_data = encoder.transform(col_data_array).ravel()
                    processed[feat] = col_data
                except Exception as e:
                    print(f"Warning: Scaling failed for '{base_col}': {e}. Using default values.")
                    processed[feat] = Config.ml.NUMERIC_IMPUTE_VALUE

            elif isinstance(encoder, LabelEncoder):
                encoded = safe_label_encode(encoder, col_data)
                processed[feat] = encoded

            elif isinstance(encoder, OneHotEncoder):
                try:
                    # Convert to numpy array and reshape for one-hot encoder
                    if hasattr(col_data, 'values'):
                        col_data_array = col_data.values.reshape(-1, 1)
                    else:
                        col_data_array = np.array(col_data).reshape(-1, 1)
                    arr = encoder.transform(col_data_array)
                    ohe_cols = [f"{base_col}_{cat}" for cat in encoder.categories_[0]]
                    arr_df = pd.DataFrame(arr, columns=ohe_cols, index=df.index)

                    for ohe_col in ohe_cols:
                        processed[ohe_col] = arr_df.get(ohe_col, 0)
                except Exception as e:
                    print(f"Warning: One-hot encoding failed for '{base_col}': {e}. Using default values.")
                    # If transformation fails, fill all expected OHE columns with 0
                    for cat in encoder.categories_[0]:
                        processed[f"{base_col}_{cat}"] = 0
        else:
            processed[feat] = col_data

    # FINAL STEP: Make sure all expected training features are present
    for feat in features:
        if feat not in processed.columns:
            print(f"Warning: Feature '{feat}' not found in processed data. Adding with default value 0.")
            processed[feat] = 0

    # Reorder columns exactly as in training
    return processed[features]
