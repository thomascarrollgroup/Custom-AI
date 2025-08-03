import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from core.config import Config

def auto_train_and_evaluate_models(X, y, prediction_type="classification", progress_callback=None):
    if prediction_type == "classification":
        return train_classification_models(X, y, progress_callback)
    else:
        return train_regression_models(X, y, progress_callback)

def train_classification_models(X, y, progress_callback=None):
    """
    Trains multiple classification models on the provided dataset and evaluates their performance.

    Parameters:
    X (array-like): Feature matrix for training.
    y (array-like): Target vector for training.
    progress_callback (callable, optional): Function to call with progress updates during model training.

    Returns:
    tuple: A tuple containing:
        - results (list of dict): A list of dictionaries with keys 'name', 'accuracy', 'f1', 'auc', and 'model',
          representing the performance and details of each trained model.
        - trained_models (dict): A dictionary of trained models with model names as keys.
    """

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=Config.ml.TEST_SIZE, random_state=Config.ml.RANDOM_STATE, stratify=y)
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000), {"C": [0.1, 1, 10]}),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=Config.ml.RANDOM_STATE), {"max_depth": [None, 5, 10]}),
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
    """
    Train multiple regression models and evaluate their performance.

    This function splits the input data into training and validation sets,
    then trains several regression models using either default parameters 
    or hyperparameter tuning with GridSearchCV. It evaluates each model's 
    performance on the validation set using Mean Squared Error (MSE) and 
    R-squared (R²) score.

    Args:
        X (array-like): Features for training and validation.
        y (array-like): Target variable for training and validation.
        progress_callback (callable, optional): A callback function to report 
            training progress with percentage completion and model name.

    Returns:
        results (list of dict): Sorted list of dictionaries containing model 
            names, their MSE, R2 scores, and trained model objects.
        trained_models (dict): Dictionary of trained models with model names 
            as keys.
    """

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=Config.ml.TEST_SIZE, random_state=Config.ml.RANDOM_STATE)
    models = [
        ("Linear Regression", LinearRegression(), {}),
        ("Ridge Regression", Ridge(), {"alpha": [0.1, 1.0, 10.0]}),
        ("Random Forest Regressor", RandomForestRegressor(n_estimators=100, random_state=Config.ml.RANDOM_STATE), {"max_depth": [None, 5, 10]}),
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
    """
    Save a trained model to a pickle file.

    Args:
        model (object): The trained model object.
        name (str): The name to use for the saved model file.
        get_writable_path (callable): A function that takes a directory name and returns a path to a writable directory.

    Returns:
        str: The file path where the model was saved.
    """
    import pickle, os
    model_dir = get_writable_path("models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    file_path = os.path.join(model_dir, f"{name}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    return file_path