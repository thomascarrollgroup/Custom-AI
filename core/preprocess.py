import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def clean_numeric(val):
    if pd.isnull(val):
        return np.nan
    val = str(val)
    match = re.findall(r'-?\d+\.?\d*', val.replace(' ', ''))
    return float(match[0]) if match else np.nan

def auto_preprocess_data(df, target_col):
    df = df.copy()
    encoders = {}
    processed_cols = []

    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            cleaned = df[col].map(clean_numeric)
            if cleaned.notnull().sum() > 0.5 * len(df):
                df[col] = cleaned
        except Exception:
            pass

    for col in df.columns:
        if col == target_col:
            if df[col].dtype == object or pd.api.types.is_categorical_dtype(df[col]):
                le = LabelEncoder()
                df[col] = df[col].astype(str).fillna("missing")
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            imputer = SimpleImputer(strategy='mean')
            df[col] = imputer.fit_transform(df[[col]]).ravel()
            encoders[col + "_imputer"] = imputer
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[[col]]).ravel()
            encoders[col] = scaler
            processed_cols.append(col)
        else:
            imputer = SimpleImputer(strategy='most_frequent')
            df[col] = imputer.fit_transform(df[[col]]).ravel()
            encoders[col + "_imputer"] = imputer
            nunique = df[col].nunique(dropna=True)
            if nunique <= 10:
                ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
                arr = ohe.fit_transform(df[[col]])
                ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                df = df.drop(columns=[col])
                df[ohe_cols] = arr
                encoders[col] = ohe
                processed_cols.extend(ohe_cols)
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
                processed_cols.append(col)

    df = df.loc[:, ~df.columns.duplicated()]
    return df, encoders, processed_cols

def preprocess_test_data(test_df, encoders, features, target_col=None):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

    df = test_df.copy()
    processed = pd.DataFrame(index=df.index)
    for feat in features:
        base_col = feat.split("_")[0]
        encoder = encoders.get(base_col)
        imputer = encoders.get(base_col + "_imputer")
        if base_col not in df.columns:
            if encoder and isinstance(encoder, OneHotEncoder):
                processed[feat] = 0
            elif encoder and isinstance(encoder, StandardScaler):
                processed[feat] = -1
            else:
                processed[feat] = -1
            continue

        col_data = df[base_col].copy()

        if imputer:
            try:
                col_data = imputer.transform(np.asarray(col_data).reshape(-1, 1)).ravel()
            except Exception:
                col_data = pd.Series(col_data).fillna(-1)

        if encoder:
            if isinstance(encoder, StandardScaler):
                col_data = encoder.transform(np.asarray(col_data).reshape(-1, 1)).ravel()
                processed[feat] = col_data
            elif isinstance(encoder, OneHotEncoder):
                arr = encoder.transform(np.asarray(col_data).reshape(-1, 1))
                ohe_cols = [f"{base_col}_{cat}" for cat in encoder.categories_[0]]
                arr_df = pd.DataFrame(arr, columns=ohe_cols, index=df.index)
                for ohe_col in ohe_cols:
                    processed[ohe_col] = arr_df[ohe_col] if ohe_col in arr_df else 0
            elif isinstance(encoder, LabelEncoder):
                known_classes = set(encoder.classes_)
                col_data = pd.Series(col_data).astype(str).apply(lambda x: x if x in known_classes else 'missing')
                if 'missing' not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, 'missing')
                col_data = encoder.transform(col_data)
                processed[feat] = col_data
            else:
                processed[feat] = col_data
        else:
            processed[feat] = col_data

    for feat in features:
        if feat not in processed.columns:
            encoder = encoders.get(feat.split("_")[0])
            if encoder and isinstance(encoder, OneHotEncoder):
                processed[feat] = 0
            else:
                processed[feat] = -1
    processed = processed[features]
    return processed