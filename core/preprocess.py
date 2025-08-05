import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, List, Any, Optional
from core.config import Config

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
    feature_to_base: Dict[str, str],   # new param
    target_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Preprocess a test DataFrame by:

    1. Handling missing values in each column
    2. Encoding categorical columns with LabelEncoder or OneHotEncoder
    3. Scaling numeric columns with StandardScaler

    Returns a DataFrame with the same features as the input DataFrame, but with
    missing values imputed and categorical columns encoded or scaled.

    Parameters:
    test_df: pd.DataFrame
        Test data to preprocess
    encoders: Dict[str, Any]
        Dictionary of encoders used for training data (from auto_preprocess_data)
    features: List[str]
        List of features to preprocess
    feature_to_base: Dict[str, str]
        Mapping of each feature to its base column (for one-hot encoded columns)
    target_col: Optional[str]
        Optional target column name for the test data (if not provided, it is
        assumed that the test data does not have a target column)

    Returns:
    pd.DataFrame
        Preprocessed test data
    """
    df = test_df.copy()
    processed = pd.DataFrame(index=df.index)

    for feat in features:
        base_col = feature_to_base.get(feat)
        if base_col is None:
            # Safety fallback: if no base_col mapping found, treat feature as base_col
            base_col = feat

        encoder = encoders.get(base_col)
        imputer = encoders.get(base_col + "_imputer")

        if base_col not in df.columns:
            # Add missing columns with sensible defaults
            if isinstance(encoder, OneHotEncoder):
                processed[feat] = 0
            elif isinstance(encoder, StandardScaler):
                processed[feat] = Config.ml.NUMERIC_IMPUTE_VALUE
            else:
                processed[feat] = Config.ml.NUMERIC_IMPUTE_VALUE
            continue

        col_data = df[base_col].copy()

        if imputer:
            try:
                col_data = imputer.transform(col_data.values.reshape(-1, 1)).ravel()
            except Exception:
                col_data = pd.Series(col_data).fillna(Config.ml.NUMERIC_IMPUTE_VALUE)

        if encoder:
            if isinstance(encoder, StandardScaler):
                col_data = encoder.transform(col_data.reshape(-1, 1)).ravel()
                processed[feat] = col_data

            elif isinstance(encoder, OneHotEncoder):
                try:
                    arr = encoder.transform(col_data.values.reshape(-1, 1))
                    ohe_cols = [f"{base_col}_{cat}" for cat in encoder.categories_[0]]
                    arr_df = pd.DataFrame(arr, columns=ohe_cols, index=df.index)
                    # Insert all one-hot columns in processed
                    for ohe_col in ohe_cols:
                        processed[ohe_col] = arr_df.get(ohe_col, 0)
                except Exception:
                    # If transform fails, create zeros for all one-hot columns
                    for cat in encoder.categories_[0]:
                        processed[f"{base_col}_{cat}"] = 0

            elif isinstance(encoder, LabelEncoder):
                encoded = safe_label_encode(encoder, col_data)
                processed[feat] = encoded

            else:
                processed[feat] = col_data
        else:
            processed[feat] = col_data

    # Make sure all required features exist (in case some were missed)
    for feat in features:
        if feat not in processed.columns:
            base_col = feature_to_base.get(feat, feat)
            enc = encoders.get(base_col)
            if isinstance(enc, OneHotEncoder):
                processed[feat] = 0
            else:
                processed[feat] = Config.ml.NUMERIC_IMPUTE_VALUE

    return processed[features]