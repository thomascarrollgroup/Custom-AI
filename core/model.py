import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    VotingClassifier, VotingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, roc_auc_score, r2_score,
    precision_score, recall_score, mean_squared_error,
    classification_report, confusion_matrix, balanced_accuracy_score
)
from sklearn.utils.multiclass import type_of_target
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

# CRITICAL FIX: Set matplotlib backend before any imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent window spawning
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

from core.config import Config

class AdvancedModelTrainer:
    """Optimized model trainer with fast algorithms, proper cross-validation, and class imbalance handling."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None
        self.feature_importance = None
        self.class_weights = None
        self.is_imbalanced = False
        
    def detect_class_imbalance(self, y):
        """Detect if the dataset has class imbalance and compute appropriate class weights."""
        unique_classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(unique_classes)
        
        if n_classes == 2:
            # Binary classification
            minority_class_count = min(class_counts)
            majority_class_count = max(class_counts)
            imbalance_ratio = minority_class_count / majority_class_count
            
            # Consider imbalanced if minority class is less than 40% of majority
            self.is_imbalanced = imbalance_ratio < 0.4
            
            if self.is_imbalanced:
                # Compute balanced class weights
                self.class_weights = compute_class_weight(
                    'balanced', 
                    classes=unique_classes, 
                    y=y
                )
                self.class_weights = dict(zip(unique_classes, self.class_weights))
                print(f"Class imbalance detected (ratio: {imbalance_ratio:.3f}). Using balanced class weights.")
            else:
                self.class_weights = None
                
        else:
            # Multi-class classification
            total_samples = len(y)
            expected_per_class = total_samples / n_classes
            min_class_count = min(class_counts)
            max_class_count = max(class_counts)
            
            # Consider imbalanced if any class has less than 60% of expected samples
            imbalance_ratio = min_class_count / expected_per_class
            self.is_imbalanced = imbalance_ratio < 0.6
            
            if self.is_imbalanced:
                # Compute balanced class weights
                self.class_weights = compute_class_weight(
                    'balanced', 
                    classes=unique_classes, 
                    y=y
                )
                self.class_weights = dict(zip(unique_classes, self.class_weights))
                print(f"Multi-class imbalance detected (min class ratio: {imbalance_ratio:.3f}). Using balanced class weights.")
            else:
                self.class_weights = None
        
        return self.is_imbalanced, self.class_weights
        
    def get_fast_preprocessing_pipeline(self, n_features=None):
        """Create a fast preprocessing pipeline."""
        if n_features and n_features > 30:
            # For high-dimensional data, use feature selection
            feature_selection = SelectKBest(k=min(30, n_features // 3))
        else:
            feature_selection = None
            
        preprocessing_steps = [
            ("imputer", SimpleImputer(strategy='median')),  # Faster than KNNImputer
            ("scaler", StandardScaler()),  # Faster than RobustScaler
        ]
        
        if feature_selection:
            preprocessing_steps.append(("feature_selection", feature_selection))
            
        return Pipeline(preprocessing_steps)
    
    def get_fast_classification_models(self):
        """Get fast classification models with optimized hyperparameters and class imbalance handling."""
        # Base models without class weights
        base_models = [
            # Random Forest - optimized for speed
            ("Random Forest", RandomForestClassifier(random_state=self.random_state), {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [10, 15, None],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__max_features": ["sqrt", "log2"]
            }),
            
            # Gradient Boosting - optimized for speed
            ("Gradient Boosting", GradientBoostingClassifier(random_state=self.random_state), {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1, 0.2],
                "classifier__max_depth": [3, 5, 7],
                "classifier__min_samples_split": [2, 5]
            }),
            
            # Extra Trees - optimized for speed
            ("Extra Trees", ExtraTreesClassifier(random_state=self.random_state), {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [10, 15, None],
                "classifier__min_samples_split": [2, 5],
                "classifier__max_features": ["sqrt", "log2"]
            }),
            
            # AdaBoost - fast and effective
            ("AdaBoost", AdaBoostClassifier(random_state=self.random_state), {
                "classifier__n_estimators": [50, 100],
                "classifier__learning_rate": [0.1, 0.5, 1.0]
            }),
            
            # Logistic Regression - fast baseline
            ("Logistic Regression", LogisticRegression(random_state=self.random_state, max_iter=1000), {
                "classifier__C": [0.1, 1, 10],
                "classifier__penalty": ["l2"],
                "classifier__solver": ["liblinear"]
            }),
            
            # K-Nearest Neighbors - fast for small datasets
            ("KNN", KNeighborsClassifier(), {
                "classifier__n_neighbors": [3, 5, 7],
                "classifier__weights": ["uniform", "distance"]
            })
        ]
        
        # If class imbalance detected, add class weights to appropriate models
        if self.is_imbalanced and self.class_weights is not None:
            enhanced_models = []
            for name, model, params in base_models:
                if name in ["Random Forest", "Extra Trees", "Logistic Regression"]:
                    # These models support class_weight parameter
                    enhanced_params = params.copy()
                    enhanced_params["classifier__class_weight"] = ["balanced"]
                    enhanced_models.append((name, model, enhanced_params))
                else:
                    # For other models, keep original params
                    enhanced_models.append((name, model, params))
            return enhanced_models
        else:
            return base_models
    
    def get_fast_regression_models(self):
        """Get fast regression models with optimized hyperparameters."""
        return [
            # Random Forest - optimized for speed
            ("Random Forest", RandomForestRegressor(random_state=self.random_state), {
                "regressor__n_estimators": [100, 200],
                "regressor__max_depth": [10, 15, None],
                "regressor__min_samples_split": [2, 5],
                "regressor__min_samples_leaf": [1, 2],
                "regressor__max_features": ["sqrt", "log2"]
            }),
            
            # Gradient Boosting - optimized for speed
            ("Gradient Boosting", GradientBoostingRegressor(random_state=self.random_state), {
                "regressor__n_estimators": [100, 200],
                "regressor__learning_rate": [0.05, 0.1, 0.2],
                "regressor__max_depth": [3, 5, 7],
                "regressor__min_samples_split": [2, 5]
            }),
            
            # Extra Trees - optimized for speed
            ("Extra Trees", ExtraTreesRegressor(random_state=self.random_state), {
                "regressor__n_estimators": [100, 200],
                "regressor__max_depth": [10, 15, None],
                "regressor__min_samples_split": [2, 5],
                "regressor__max_features": ["sqrt", "log2"]
            }),
            
            # AdaBoost - fast and effective
            ("AdaBoost", AdaBoostRegressor(random_state=self.random_state), {
                "regressor__n_estimators": [50, 100],
                "regressor__learning_rate": [0.1, 0.5, 1.0]
            }),
            
            # Ridge Regression - fast baseline
            ("Ridge Regression", Ridge(random_state=self.random_state), {
                "regressor__alpha": [0.1, 1, 10, 100]
            }),
            
            # K-Nearest Neighbors - fast for small datasets
            ("KNN", KNeighborsRegressor(), {
                "regressor__n_neighbors": [3, 5, 7],
                "regressor__weights": ["uniform", "distance"]
            })
        ]
    
    def create_simple_ensemble(self, base_models, task_type="classification"):
        """Create a simple voting ensemble for speed."""
        if task_type == "classification":
            return VotingClassifier(
                estimators=[(name, model) for name, model in base_models],
                voting='soft'
            )
        else:
            return VotingRegressor(
                estimators=[(name, model) for name, model in base_models]
            )
    
    def get_appropriate_scoring(self, task_type="classification"):
        """Get appropriate scoring metric based on task type and class imbalance."""
        if task_type == "classification":
            if self.is_imbalanced:
                # Use balanced accuracy and F1 for imbalanced datasets
                return 'balanced_accuracy'
            else:
                # Use regular accuracy for balanced datasets
                return 'accuracy'
        else:
            return 'r2'
    
    def evaluate_with_cv(self, model, X, y, task_type="classification", cv_folds=3):
        """Evaluate model with cross-validation to detect overfitting/underfitting."""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state) if task_type == "classification" else cv_folds
        
        # Use appropriate scoring metric
        scoring = self.get_appropriate_scoring(task_type)
        
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def compute_balanced_metrics(self, y_true, y_pred, y_proba=None):
        """Compute comprehensive metrics for classification, especially for imbalanced datasets."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # F1 scores
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        
        # Precision and Recall
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # AUC if probabilities available
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class classification
                    metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            except Exception:
                metrics['auc'] = None
        else:
            metrics['auc'] = None
        
        return metrics
    
    def train_with_optimized_optimization(self, X, y, task_type="classification", progress_callback=None):
        """Train models with optimized hyperparameter search, cross-validation, and class imbalance handling."""
        print('train:', X.columns)
        print('test:', y.name)
        
        # Detect class imbalance for classification tasks
        if task_type == "classification":
            self.detect_class_imbalance(y)
        
        # Split data with stratification for classification
        if task_type == "classification":
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )
        
        # Get fast preprocessing pipeline
        preprocessing = self.get_fast_preprocessing_pipeline(X.shape[1])
        
        # Get fast models (with class imbalance handling for classification)
        if task_type == "classification":
            models = self.get_fast_classification_models()
        else:
            models = self.get_fast_regression_models()
        
        results = []
        trained_models = {}
        total_models = len(models)
        
        # Get appropriate scoring metric
        scoring = self.get_appropriate_scoring(task_type)
        
        # Train individual models with cross-validation
        for idx, (name, model, params) in enumerate(models):
            try:
                pipeline = Pipeline([
                    ("preprocessing", preprocessing),
                    ("classifier" if task_type == "classification" else "regressor", model)
                ])
                
                # Use RandomizedSearchCV with fewer iterations for speed
                if params:
                    search = RandomizedSearchCV(
                        pipeline, params, 
                        n_iter=10,  # Reduced from 20 for speed
                        cv=3,  # Reduced from 5 for speed
                        n_jobs=1, 
                        random_state=self.random_state,
                        scoring=scoring
                    )
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                    best_score = search.best_score_
                else:
                    best_model = pipeline
                    best_model.fit(X_train, y_train)
                    best_params = {}
                    best_score = None
                
                # Evaluate with cross-validation
                cv_results = self.evaluate_with_cv(best_model, X_train, y_train, task_type, cv_folds=3)
                
                # Evaluate on validation set
                y_pred = best_model.predict(X_val)
                
                if task_type == "classification":
                    # Get probabilities if available
                    y_proba = None
                    try:
                        if hasattr(best_model.named_steps["classifier"], "predict_proba"):
                            y_proba = best_model.predict_proba(X_val)
                    except Exception:
                        pass
                    
                    # Compute comprehensive metrics
                    metrics = self.compute_balanced_metrics(y_val, y_pred, y_proba)
                    
                    results.append({
                        "name": name,
                        "accuracy": metrics['accuracy'],
                        "f1": metrics['f1_weighted'],  # Keep for backward compatibility
                        "precision": metrics['precision_weighted'],  # Keep for backward compatibility
                        "recall": metrics['recall_weighted'],  # Keep for backward compatibility
                        "auc": metrics['auc'],
                        "balanced_accuracy": metrics['balanced_accuracy'],
                        "f1_macro": metrics['f1_macro'],
                        "f1_micro": metrics['f1_micro'],
                        "precision_macro": metrics['precision_macro'],
                        "recall_macro": metrics['recall_macro'],
                        "model": best_model,
                        "best_params": best_params,
                        "cv_mean": cv_results['cv_mean'],
                        "cv_std": cv_results['cv_std'],
                        "cv_scores": cv_results['cv_scores'],
                        "best_cv_score": best_score
                    })
                else:
                    mae = mean_absolute_error(y_val, y_pred)
                    mse = mean_squared_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    
                    results.append({
                        "name": name,
                        "mae": mae,
                        "mse": mse,
                        "r2": r2,
                        "model": best_model,
                        "best_params": best_params,
                        "cv_mean": cv_results['cv_mean'],
                        "cv_std": cv_results['cv_std'],
                        "cv_scores": cv_results['cv_scores'],
                        "best_cv_score": best_score
                    })
                
                trained_models[name] = best_model
                
                if progress_callback:
                    percent = int((idx + 1) / total_models * 90)  # Reserve 10% for ensemble
                    progress_callback(percent, f"Training {name}")
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        # Create simple voting ensemble with top 2 models
        if len(trained_models) >= 2:
            try:
                if progress_callback:
                    progress_callback(95, "Creating voting ensemble")
                
                # Get top 2 models for ensemble (use balanced accuracy for imbalanced datasets)
                if task_type == "classification" and self.is_imbalanced:
                    top_models = sorted(results, key=lambda x: x["balanced_accuracy"], reverse=True)[:2]
                else:
                    top_models = sorted(results, 
                                      key=lambda x: x["accuracy" if task_type == "classification" else "r2"], 
                                      reverse=True)[:2]
                
                ensemble_model = self.create_simple_ensemble(
                    [(r["name"], r["model"]) for r in top_models],
                    task_type
                )
                
                # Train ensemble model
                ensemble_model.fit(X_train, y_train)
                y_pred = ensemble_model.predict(X_val)
                
                # Evaluate ensemble with cross-validation
                cv_results = self.evaluate_with_cv(ensemble_model, X_train, y_train, task_type, cv_folds=3)
                
                if task_type == "classification":
                    # Get probabilities if available
                    y_proba = None
                    try:
                        if hasattr(ensemble_model, "predict_proba"):
                            y_proba = ensemble_model.predict_proba(X_val)
                    except Exception:
                        pass
                    
                    # Compute comprehensive metrics
                    metrics = self.compute_balanced_metrics(y_val, y_pred, y_proba)
                    
                    results.append({
                        "name": "Voting Ensemble",
                        "accuracy": metrics['accuracy'],
                        "f1": metrics['f1_weighted'],
                        "precision": metrics['precision_weighted'],
                        "recall": metrics['recall_weighted'],
                        "auc": metrics['auc'],
                        "balanced_accuracy": metrics['balanced_accuracy'],
                        "f1_macro": metrics['f1_macro'],
                        "f1_micro": metrics['f1_micro'],
                        "precision_macro": metrics['precision_macro'],
                        "recall_macro": metrics['recall_macro'],
                        "model": ensemble_model,
                        "best_params": {},
                        "cv_mean": cv_results['cv_mean'],
                        "cv_std": cv_results['cv_std'],
                        "cv_scores": cv_results['cv_scores'],
                        "best_cv_score": None
                    })
                else:
                    mae = mean_absolute_error(y_val, y_pred)
                    mse = mean_squared_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    
                    results.append({
                        "name": "Voting Ensemble",
                        "mae": mae,
                        "mse": mse,
                        "r2": r2,
                        "model": ensemble_model,
                        "best_params": {},
                        "cv_mean": cv_results['cv_mean'],
                        "cv_std": cv_results['cv_std'],
                        "cv_scores": cv_results['cv_scores'],
                        "best_cv_score": None
                    })
                
                trained_models["Voting Ensemble"] = ensemble_model
                
            except Exception as e:
                print(f"Error creating ensemble: {e}")
        
        # Sort results by performance (use balanced accuracy for imbalanced classification)
        if task_type == "classification":
            if self.is_imbalanced:
                results.sort(key=lambda r: r["balanced_accuracy"], reverse=True)
            else:
                results.sort(key=lambda r: r["accuracy"], reverse=True)
        else:
            results.sort(key=lambda r: r["r2"], reverse=True)
        
        # Store best model and feature importance
        if results:
            self.best_model = results[0]["model"]
            # Try to get feature importance from the best model
            try:
                if hasattr(self.best_model.named_steps["classifier"], "feature_importances_"):
                    self.feature_importance = self.best_model.named_steps["classifier"].feature_importances_
                elif hasattr(self.best_model.named_steps["regressor"], "feature_importances_"):
                    self.feature_importance = self.best_model.named_steps["regressor"].feature_importances_
            except:
                self.feature_importance = None
        
        if progress_callback:
            progress_callback(100, "Training complete")
        
        return results, trained_models

def auto_train_and_evaluate_models(X, y, prediction_type="classification", progress_callback=None):
    """Main function to train and evaluate models with optimized algorithms and class imbalance handling."""
    trainer = AdvancedModelTrainer(random_state=Config.ml.RANDOM_STATE)
    return trainer.train_with_optimized_optimization(X, y, prediction_type, progress_callback)

def save_model(model, name, get_writable_path):
    """Save a trained model to disk."""
    import pickle, os
    model_dir = get_writable_path("models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    file_path = os.path.join(model_dir, f"{name}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    return file_path

# Legacy functions for backward compatibility
def train_classification_models(X, y, progress_callback=None):
    """Legacy function - now uses optimized trainer with class imbalance handling."""
    return auto_train_and_evaluate_models(X, y, "classification", progress_callback)

def train_regression_models(X, y, progress_callback=None):
    """Legacy function - now uses optimized trainer."""
    return auto_train_and_evaluate_models(X, y, "regression", progress_callback)
