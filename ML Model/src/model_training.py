"""
Model Training Module
Handles Random Forest model training with hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Class for training and tuning Random Forest model"""
    
    def __init__(self, X, y):
        """
        Initialize the ModelTrainer
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        self.X = X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.best_model = None
        self.best_params = None
        self.feature_importance = None
        
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"Data split complete:")
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_baseline_model(self):
        """Train a baseline Random Forest model"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate baseline
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\nBaseline Random Forest model trained. Test accuracy: {accuracy:.4f}")
        
        return self.model

    def train_xgboost_model(self):
        """Train a baseline XGBoost model"""
        self.model = XGBClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate baseline
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\nBaseline XGBoost model trained. Test accuracy: {accuracy:.4f}")
        
        return self.model
    
    def hyperparameter_tuning(self, cv=5, n_jobs=-1):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
        """
        print("\n=== Starting Hyperparameter Tuning ===")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Initialize base model
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=cv,
            scoring='f1_macro',
            n_jobs=n_jobs,
            verbose=1
        )
        
        print("Fitting GridSearchCV...")
        grid_search.fit(self.X_train, self.y_train)
        
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"\n=== Hyperparameter Tuning Complete ===")
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self.best_model, self.best_params
    
    def randomized_hyperparameter_tuning(self, n_iter=30, cv=3, n_jobs=-1):
        """
        Perform hyperparameter tuning using RandomizedSearchCV for Random Forest
        """
        print("\n=== Starting Randomized Hyperparameter Tuning (Random Forest) ===")
        
        # Initialize base model
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Take a small random subset for tuning
        X_sub, _, y_sub, _ = train_test_split(self.X_train, self.y_train, train_size=50000, random_state=42, stratify=self.y_train)
        
        # Define parameter distribution
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Perform random search
        rand_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='f1_macro',
            n_jobs=n_jobs,
            verbose=2,
            random_state=42
        )
        
        print("Fitting RandomizedSearchCV...")
        rand_search.fit(X_sub, y_sub)
        
        self.best_model = rand_search.best_estimator_
        self.best_params = rand_search.best_params_
        
        print(f"\n=== Randomized Hyperparameter Tuning Complete ===")
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {rand_search.best_score_:.4f}")
        
        return self.best_model, self.best_params

    def randomized_hyperparameter_tuning_xgboost(self, n_iter=30, cv=3, n_jobs=-1):
        """
        Perform hyperparameter tuning using RandomizedSearchCV for XGBoost
        """
        print("\n=== Starting Randomized Hyperparameter Tuning (XGBoost) ===")
        
        # Initialize base model
        xgb = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
        
        # Take a small random subset for tuning
        X_sub, _, y_sub, _ = train_test_split(self.X_train, self.y_train, train_size=50000, random_state=42, stratify=self.y_train)
        
        # Define parameter distribution
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
        
        # Perform random search
        rand_search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='f1_macro',
            n_jobs=n_jobs,
            verbose=2,
            random_state=42
        )
        
        print("Fitting RandomizedSearchCV (XGBoost)...")
        rand_search.fit(X_sub, y_sub)
        
        self.best_model = rand_search.best_estimator_
        self.best_params = rand_search.best_params_
        
        print(f"\n=== Randomized Hyperparameter Tuning Complete (XGBoost) ===")
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {rand_search.best_score_:.4f}")
        
        return self.best_model, self.best_params
    
    def evaluate_model(self, model=None):
        """
        Evaluate the model on test set
        
        Args:
            model: Model to evaluate. If None, use best_model
        """
        if model is None:
            model = self.best_model if self.best_model is not None else self.model
        
        if model is None:
            raise ValueError("No model available for evaluation. Please train a model first.")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n=== Model Evaluation Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"\nConfusion Matrix:\n{cm}")
        print(f"\nTop 10 Most Important Features:")
        print(self.feature_importance.head(10))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'feature_importance': self.feature_importance,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def cross_validate(self, model=None, cv=5):
        """
        Perform cross-validation
        
        Args:
            model: Model to evaluate. If None, use best_model
            cv: Number of folds
        """
        if model is None:
            model = self.best_model if self.best_model is not None else self.model
        
        if model is None:
            raise ValueError("No model available for cross-validation.")
        
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1_macro')
        
        print(f"\n=== Cross-Validation Results ({cv}-fold) ===")
        print(f"Mean F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Individual fold scores: {cv_scores}")
        
        return cv_scores
    
    def save_model(self, filepath='models/accident_severity_model.pkl'):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        import os
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        model_to_save = self.best_model if self.best_model is not None else self.model
        
        if model_to_save is None:
            raise ValueError("No model available to save.")
        
        joblib.dump(model_to_save, filepath)
        print(f"Model saved to {filepath}")
        
        return filepath

