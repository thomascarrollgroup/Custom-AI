import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, roc_auc_score, r2_score,
    precision_score, recall_score, mean_squared_error, log_loss,
    classification_report, confusion_matrix
)
from sklearn.compose import ColumnTransformer
from sklearn.utils.multiclass import type_of_target
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.decomposition import PCA
from scipy.stats import randint, uniform, loguniform
import warnings
warnings.filterwarnings('ignore')

# CRITICAL FIX: Set matplotlib backend before any imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent window spawning
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

from core.config import Config

class AdvancedModelTrainer:
    """Advanced model trainer with state-of-the-art algorithms and ensemble methods."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None
        self.feature_importance = None
        
    def get_advanced_preprocessing_pipeline(self, n_features=None):
        """Create an advanced preprocessing pipeline with multiple strategies."""
        if n_features and n_features > 50:
            # For high-dimensional data, use feature selection
            feature_selection = SelectKBest(k=min(50, n_features // 2))
        else:
            feature_selection = None
            
        preprocessing_steps = [
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", RobustScaler()),  # More robust than StandardScaler
        ]
        
        if feature_selection:
            preprocessing_steps.append(("feature_selection", feature_selection))
            
        return Pipeline(preprocessing_steps)
    
    def get_classification_models(self):
        """Get a comprehensive set of classification models with advanced hyperparameters."""
        return [
            # Advanced Random Forest
            ("Advanced Random Forest", RandomForestClassifier(random_state=self.random_state), {
                "classifier__n_estimators": [200, 300, 500],
                "classifier__max_depth": [None, 15, 20, 25],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", "log2", None],
                "classifier__bootstrap": [True, False],
                "classifier__class_weight": ["balanced", "balanced_subsample", None]
            }),
            
            # Gradient Boosting
            ("Gradient Boosting", GradientBoostingClassifier(random_state=self.random_state), {
                "classifier__n_estimators": [200, 300, 500],
                "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "classifier__max_depth": [3, 5, 7, 9],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__subsample": [0.8, 0.9, 1.0]
            }),
            
            # Extra Trees
            ("Extra Trees", ExtraTreesClassifier(random_state=self.random_state), {
                "classifier__n_estimators": [200, 300, 500],
                "classifier__max_depth": [None, 15, 20, 25],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", "log2", None],
                "classifier__bootstrap": [True, False]
            }),
            
            # Support Vector Machine
            ("SVM", SVC(random_state=self.random_state, probability=True), {
                "classifier__C": [0.1, 1, 10, 100],
                "classifier__kernel": ["rbf", "poly", "sigmoid"],
                "classifier__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                "classifier__class_weight": ["balanced", None]
            }),
            
            # Neural Network
            ("Neural Network", MLPClassifier(random_state=self.random_state, max_iter=1000), {
                "classifier__hidden_layer_sizes": [(100,), (100, 50), (200, 100), (100, 50, 25)],
                "classifier__activation": ["relu", "tanh"],
                "classifier__alpha": [0.0001, 0.001, 0.01],
                "classifier__learning_rate": ["constant", "adaptive"],
                "classifier__learning_rate_init": [0.001, 0.01, 0.1]
            }),
            
            # AdaBoost
            ("AdaBoost", AdaBoostClassifier(random_state=self.random_state), {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__learning_rate": [0.01, 0.1, 0.5, 1.0],
                "classifier__algorithm": ["SAMME", "SAMME.R"]
            }),
            
            # Logistic Regression with advanced regularization
            ("Advanced Logistic Regression", LogisticRegression(random_state=self.random_state, max_iter=2000), {
                "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "classifier__penalty": ["l1", "l2", "elasticnet"],
                "classifier__solver": ["liblinear", "saga"],
                "classifier__class_weight": ["balanced", None]
            }),
            
            # K-Nearest Neighbors
            ("KNN", KNeighborsClassifier(), {
                "classifier__n_neighbors": [3, 5, 7, 9, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2],
                "classifier__leaf_size": [10, 20, 30]
            })
        ]
    
    def get_regression_models(self):
        """Get a comprehensive set of regression models with advanced hyperparameters."""
        return [
            # Advanced Random Forest
            ("Advanced Random Forest", RandomForestRegressor(random_state=self.random_state), {
                "regressor__n_estimators": [200, 300, 500],
                "regressor__max_depth": [None, 15, 20, 25],
                "regressor__min_samples_split": [2, 5, 10],
                "regressor__min_samples_leaf": [1, 2, 4],
                "regressor__max_features": ["sqrt", "log2", None],
                "regressor__bootstrap": [True, False]
            }),
            
            # Gradient Boosting
            ("Gradient Boosting", GradientBoostingRegressor(random_state=self.random_state), {
                "regressor__n_estimators": [200, 300, 500],
                "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "regressor__max_depth": [3, 5, 7, 9],
                "regressor__min_samples_split": [2, 5, 10],
                "regressor__min_samples_leaf": [1, 2, 4],
                "regressor__subsample": [0.8, 0.9, 1.0]
            }),
            
            # Extra Trees
            ("Extra Trees", ExtraTreesRegressor(random_state=self.random_state), {
                "regressor__n_estimators": [200, 300, 500],
                "regressor__max_depth": [None, 15, 20, 25],
                "regressor__min_samples_split": [2, 5, 10],
                "regressor__min_samples_leaf": [1, 2, 4],
                "regressor__max_features": ["sqrt", "log2", None],
                "regressor__bootstrap": [True, False]
            }),
            
            # Support Vector Regression
            ("SVR", SVR(), {
                "regressor__C": [0.1, 1, 10, 100],
                "regressor__kernel": ["rbf", "poly", "linear"],
                "regressor__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                "regressor__epsilon": [0.01, 0.1, 0.2]
            }),
            
            # Neural Network
            ("Neural Network", MLPRegressor(random_state=self.random_state, max_iter=1000), {
                "regressor__hidden_layer_sizes": [(100,), (100, 50), (200, 100), (100, 50, 25)],
                "regressor__activation": ["relu", "tanh"],
                "regressor__alpha": [0.0001, 0.001, 0.01],
                "regressor__learning_rate": ["constant", "adaptive"],
                "regressor__learning_rate_init": [0.001, 0.01, 0.1]
            }),
            
            # AdaBoost
            ("AdaBoost", AdaBoostRegressor(random_state=self.random_state), {
                "regressor__n_estimators": [50, 100, 200],
                "regressor__learning_rate": [0.01, 0.1, 0.5, 1.0],
                "regressor__loss": ["linear", "square", "exponential"]
            }),
            
            # Elastic Net
            ("Elastic Net", ElasticNet(random_state=self.random_state), {
                "regressor__alpha": [0.001, 0.01, 0.1, 1, 10],
                "regressor__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                "regressor__max_iter": [2000, 3000]
            }),
            
            # K-Nearest Neighbors
            ("KNN", KNeighborsRegressor(), {
                "regressor__n_neighbors": [3, 5, 7, 9, 11],
                "regressor__weights": ["uniform", "distance"],
                "regressor__p": [1, 2],
                "regressor__leaf_size": [10, 20, 30]
            })
        ]
    
    def create_ensemble_models(self, base_models, X_train, y_train, task_type="classification"):
        """Create powerful ensemble models using voting and stacking."""
        if task_type == "classification":
            # Voting Classifier
            voting_clf = VotingClassifier(
                estimators=[(name, model) for name, model in base_models],
                voting='soft'
            )
            
            # Stacking Classifier with meta-learner
            meta_learner = LogisticRegression(random_state=self.random_state)
            stacking_clf = StackingClassifier(
                estimators=[(name, model) for name, model in base_models],
                final_estimator=meta_learner,
                cv=5
            )
            
            return {
                "Voting Ensemble": voting_clf,
                "Stacking Ensemble": stacking_clf
            }
        else:
            # Voting Regressor
            voting_reg = VotingRegressor(
                estimators=[(name, model) for name, model in base_models]
            )
            
            # Stacking Regressor with meta-learner
            meta_learner = Ridge(random_state=self.random_state)
            stacking_reg = StackingRegressor(
                estimators=[(name, model) for name, model in base_models],
                final_estimator=meta_learner,
                cv=5
            )
            
            return {
                "Voting Ensemble": voting_reg,
                "Stacking Ensemble": stacking_reg
            }
    
    def train_with_advanced_optimization(self, X, y, task_type="classification", progress_callback=None):
        """Train models with advanced hyperparameter optimization."""
        # Split data with stratification for classification
        if task_type == "classification":
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )
        
        # Get preprocessing pipeline
        preprocessing = self.get_advanced_preprocessing_pipeline(X.shape[1])
        
        # Get models
        if task_type == "classification":
            models = self.get_classification_models()
        else:
            models = self.get_regression_models()
        
        results = []
        trained_models = {}
        total_models = len(models)
        
        # Train individual models
        for idx, (name, model, params) in enumerate(models):
            try:
                pipeline = Pipeline([
                    ("preprocessing", preprocessing),
                    ("classifier" if task_type == "classification" else "regressor", model)
                ])
                
                # Use RandomizedSearchCV for faster optimization
                if params:
                    search = RandomizedSearchCV(
                        pipeline, params, 
                        n_iter=20,  # More iterations for better optimization
                        cv=5, 
                        n_jobs=1, 
                        random_state=self.random_state,
                        scoring='accuracy' if task_type == "classification" else 'r2'
                    )
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                else:
                    best_model = pipeline
                    best_model.fit(X_train, y_train)
                    best_params = {}
                
                # Evaluate model
                y_pred = best_model.predict(X_val)
                
                if task_type == "classification":
                    acc = accuracy_score(y_val, y_pred)
                    f1 = f1_score(y_val, y_pred, average="weighted")
                    precision = precision_score(y_val, y_pred, average="weighted")
                    recall = recall_score(y_val, y_pred, average="weighted")
                    
                    # Calculate AUC if possible
                    auc = None
                    try:
                        if hasattr(best_model.named_steps["classifier"], "predict_proba"):
                            y_proba = best_model.predict_proba(X_val)
                            if type_of_target(y_val) == "binary":
                                auc = roc_auc_score(y_val, y_proba[:, 1])
                            elif y_proba.ndim == 2 and y_proba.shape[1] == len(np.unique(y_val)):
                                auc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')
                    except Exception:
                        auc = None
                    
                    results.append({
                        "name": name,
                        "accuracy": acc,
                        "f1": f1,
                        "precision": precision,
                        "recall": recall,
                        "auc": auc,
                        "model": best_model,
                        "best_params": best_params
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
                        "best_params": best_params
                    })
                
                trained_models[name] = best_model
                
                if progress_callback:
                    percent = int((idx + 1) / total_models * 80)  # Reserve 20% for ensembles
                    progress_callback(percent, f"Training {name}")
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        # Create and train ensemble models
        if len(trained_models) >= 2:
            try:
                if progress_callback:
                    progress_callback(85, "Creating ensemble models")
                
                # Get top 3 models for ensemble
                top_models = sorted(results, 
                                  key=lambda x: x["accuracy" if task_type == "classification" else "r2"], 
                                  reverse=True)[:3]
                
                ensemble_models = self.create_ensemble_models(
                    [(r["name"], r["model"]) for r in top_models],
                    X_train, y_train, task_type
                )
                
                # Train ensemble models
                for name, ensemble_model in ensemble_models.items():
                    try:
                        ensemble_model.fit(X_train, y_train)
                        y_pred = ensemble_model.predict(X_val)
                        
                        if task_type == "classification":
                            acc = accuracy_score(y_val, y_pred)
                            f1 = f1_score(y_val, y_pred, average="weighted")
                            precision = precision_score(y_val, y_pred, average="weighted")
                            recall = recall_score(y_val, y_pred, average="weighted")
                            
                            results.append({
                                "name": name,
                                "accuracy": acc,
                                "f1": f1,
                                "precision": precision,
                                "recall": recall,
                                "auc": None,  # Ensembles might not have predict_proba
                                "model": ensemble_model,
                                "best_params": {}
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
                                "model": ensemble_model,
                                "best_params": {}
                            })
                        
                        trained_models[name] = ensemble_model
                        
                    except Exception as e:
                        print(f"Error training ensemble {name}: {e}")
                        continue
                
                if progress_callback:
                    progress_callback(95, "Ensemble training complete")
                    
            except Exception as e:
                print(f"Error creating ensembles: {e}")
        
        # Sort results by performance
        if task_type == "classification":
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
    """Main function to train and evaluate models with advanced algorithms."""
    trainer = AdvancedModelTrainer(random_state=Config.ml.RANDOM_STATE)
    return trainer.train_with_advanced_optimization(X, y, prediction_type, progress_callback)

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
    """Legacy function - now uses advanced trainer."""
    return auto_train_and_evaluate_models(X, y, "classification", progress_callback)

def train_regression_models(X, y, progress_callback=None):
    """Legacy function - now uses advanced trainer."""
    return auto_train_and_evaluate_models(X, y, "regression", progress_callback)
