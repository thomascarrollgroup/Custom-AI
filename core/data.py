import pandas as pd
import os

def load_csv(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")