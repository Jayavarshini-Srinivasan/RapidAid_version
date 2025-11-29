"""
Enhanced Model Tuning Module
Implements hyperparameter optimization for:
- Random Forest models
- Gradient Boosting models
Uses GridSearchCV and RandomizedSearchCV with 5-fold cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# Optional LightGBM (faster for large datasets). Import if available.
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    lgb_available = True
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None
    lgb_available = False

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import make_scorer, f1_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')


class EnhancedModelTuner:
    """Enhanced model tuning with comprehensive hyperparameter optimization"""
    
    def __init__(self, X, y, task_type='classification', random_state=42):
        """
        Initialize the EnhancedModelTuner
        
        Args:
            X: Feature matrix
            y: Target vector
            task_type: 'classification' or 'regression'
            random_state: Random seed for reproducibility
        """
        self.X = X
        self.y = y
        self.task_type = task_type
        self.random_state = random_state
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.rf_model = None
        self.gb_model = None
        self.rf_best_params = None
        self.gb_best_params = None
        self.rf_best_score = None
        self.gb_best_score = None
        
    def split_data(self, test_size=0.2):
        """Split data into training and testing sets"""
        if self.task_type == 'classification':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=self.random_state, 
                stratify=self.y
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=self.random_state
            )
        
        print(f"Data split complete:")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Testing set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    # ==================== RANDOM FOREST TUNING ====================
    
    def tune_random_forest(self, method='grid', cv=5, n_jobs=-1, verbose=1):
        """
        Tune Random Forest hyperparameters
        
        Args:
            method: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
        
        Returns:
            Best model and parameters
        """
        print("\n" + "="*60)
        print("RANDOM FOREST HYPERPARAMETER TUNING")
        print("="*60)
        
        # Select appropriate model based on task type
        if self.task_type == 'classification':
            base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=n_jobs)
            scoring = 'f1_macro'  # F1-score for classification
        else:
            base_model = RandomForestRegressor(random_state=self.random_state, n_jobs=n_jobs)
            scoring = 'neg_mean_squared_error'  # MSE for regression
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Select CV strategy
        if self.task_type == 'classification':
            cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        if method == 'grid':
            print(f"Using GridSearchCV with {cv}-fold cross-validation...")
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv_strategy,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                return_train_score=True
            )
        else:  # randomized
            print(f"Using RandomizedSearchCV with {cv}-fold cross-validation...")
            n_iter = 50  # Number of parameter settings to sample
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv_strategy,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=self.random_state,
                return_train_score=True
            )
        
        print("Fitting model...")
        search.fit(self.X_train, self.y_train)
        
        self.rf_model = search.best_estimator_
        self.rf_best_params = search.best_params_
        self.rf_best_score = search.best_score_
        
        print(f"\n✓ Random Forest tuning complete!")
        print(f"  Best parameters: {self.rf_best_params}")
        print(f"  Best CV score: {self.rf_best_score:.4f}")
        
        # Display top 5 parameter combinations
        results_df = pd.DataFrame(search.cv_results_)
        top_results = results_df.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
        print(f"\n  Top 5 parameter combinations:")
        for idx, row in top_results.iterrows():
            print(f"    Score: {row['mean_test_score']:.4f} (+/- {row['std_test_score']*2:.4f})")
            print(f"    Params: {row['params']}")
        
        return self.rf_model, self.rf_best_params
    
    # ==================== GRADIENT BOOSTING TUNING ====================
    
    def tune_gradient_boosting(self, method='grid', cv=5, n_jobs=-1, verbose=1):
        """
        Tune Gradient Boosting hyperparameters
        
        Args:
            method: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
        
        Returns:
            Best model and parameters
        """
        print("\n" + "="*60)
        print("GRADIENT BOOSTING HYPERPARAMETER TUNING")
        print("="*60)
        
        # Prefer LightGBM if available (faster and scalable). Otherwise use HistGradientBoosting.
        if lgb_available:
            if self.task_type == 'classification':
                base_model = LGBMClassifier(random_state=self.random_state, n_jobs=n_jobs)
                scoring = 'f1_macro'
            else:
                base_model = LGBMRegressor(random_state=self.random_state, n_jobs=n_jobs)
                scoring = 'neg_mean_squared_error'

