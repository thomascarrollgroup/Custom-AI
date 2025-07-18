import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score

def auto_train_and_evaluate_models(X, y, prediction_type="classification", progress_callback=None):
    if prediction_type == "classification":
        return train_classification_models(X, y, progress_callback)
    else:
        return train_regression_models(X, y, progress_callback)

def train_classification_models(X, y, progress_callback=None):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000), {"C": [0.1, 1, 10]}),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42), {"max_depth": [None, 5, 10]}),
        ("Gaussian Naive Bayes", GaussianNB(), {}),
        ("K-Nearest Neighbors", KNeighborsClassifier(), {"n_neighbors": [3, 5, 7]})
    ]
    results = []
    trained_models = {}
    total = len(models)
    for idx, (name, model, params) in enumerate(models):
        if params:
            grid = GridSearchCV(model, params, cv=3, n_jobs=1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
        else:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")
        auc = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)
            if len(np.unique(y_val)) == 2:
                auc = roc_auc_score(y_val, y_proba[:, 1])
            elif y_proba.ndim == 2 and y_proba.shape[1] == len(np.unique(y_val)):
                auc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')
        results.append({
            "name": name,
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "model": model
        })
        trained_models[name] = model
        if progress_callback:
            percent = int((idx + 1) / total * 100)
            progress_callback(percent, name)
    results.sort(key=lambda r: r["accuracy"], reverse=True)
    return results, trained_models

def train_regression_models(X, y, progress_callback=None):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    models = [
        ("Linear Regression", LinearRegression(), {}),
        ("Ridge Regression", Ridge(), {"alpha": [0.1, 1.0, 10.0]}),
        ("Random Forest Regressor", RandomForestRegressor(n_estimators=100, random_state=42), {"max_depth": [None, 5, 10]}),
        ("K-Nearest Neighbors Regressor", KNeighborsRegressor(), {"n_neighbors": [3, 5, 7]})
    ]
    results = []
    trained_models = {}
    total = len(models)
    for idx, (name, model, params) in enumerate(models):
        if params:
            grid = GridSearchCV(model, params, cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
        else:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        results.append({
            "name": name,
            "mse": mse,
            "r2": r2,
            "model": model
        })
        trained_models[name] = model
        if progress_callback:
            percent = int((idx + 1) / total * 100)
            progress_callback(percent, name)
    results.sort(key=lambda r: r["r2"], reverse=True)
    return results, trained_models

def save_model(model, name, get_writable_path):
    import pickle, os
    model_dir = get_writable_path("models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    file_path = os.path.join(model_dir, f"{name}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    return file_path