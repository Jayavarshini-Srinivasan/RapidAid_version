"""
Simple Script to Run Models
This script provides a simple interface to run the ML pipeline
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import our modules
try:
    from src.feature_engineering.enhanced_feature_engineering import EnhancedFeatureEngineer
    from src.model_training.enhanced_model_tuning import EnhancedModelTuner
    from src.evaluation.comprehensive_evaluation import ComprehensiveEvaluator
    from src.visualization.visualization_integration import VisualizationDashboard
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure you are running this script from the correct environment and directory structure.")
    sys.exit(1)


def load_data(file_path=None):
    """Load data from file or use sample data"""
    if file_path and os.path.exists(file_path):
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path, nrows=50000)  # Limit for demo
        print(f"✓ Loaded {len(df)} records")
        return df
    else:
        # Try to find data files in data directory
        data_dir = PROJECT_ROOT / 'data'
        data_files = []
        if data_dir.exists():
            # Look for dispatch data or accident data
            data_files = [str(f) for f in data_dir.glob('*.csv') 
                         if 'synthetic_dispatch' in f.name.lower() or 'accident' in f.name.lower()]
        
        if data_files:
            # Prefer synthetic_dispatch if available, else take the first one
            dispatch_files = [f for f in data_files if 'synthetic_dispatch' in f.lower()]
            selected_file = dispatch_files[0] if dispatch_files else data_files[0]
            
            print(f"Found data file: {selected_file}")
            df = pd.read_csv(selected_file, nrows=50000)
            print(f"✓ Loaded {len(df)} records")
            return df
        else:
            print("No data file found. Creating sample data...")
            return create_sample_data()


def create_sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='1H')
    df = pd.DataFrame({
        'timestamp': dates,
        'latitude': np.random.normal(12.9716, 0.1, n_samples),
        'longitude': np.random.normal(77.5946, 0.1, n_samples),
        'severity': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.5, 0.3, 0.2]),
        'acceleration_x': np.random.normal(0, 1, n_samples),
        'acceleration_y': np.random.normal(0, 1, n_samples),
        'estimated_response_time': np.random.normal(10, 3, n_samples)
    })
    print(f"✓ Created {len(df)} sample records")
    return df


def run_pipeline(df, task_type='classification', quick_mode=True):
    """
    Run the complete ML pipeline
    
    Args:
        df: DataFrame with data
        task_type: 'classification' or 'regression'
        quick_mode: If True, use faster settings (random search, fewer CV folds)
    """
    print("\n" + "="*80)
    print("RUNNING ML PIPELINE")
    print("="*80)
    
    # Ensure output directories exist
    output_dir = PROJECT_ROOT / 'output'
    models_dir = PROJECT_ROOT / 'models'
    output_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    # Step 1: Feature Engineering
    print("\n[1/5] Feature Engineering...")
    engineer = EnhancedFeatureEngineer(df)
    
    # Detect columns
    timestamp_col = None
    lat_col = None
    lon_col = None
    severity_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'timestamp' in col_lower or 'datetime' in col_lower or 'date' in col_lower:
            timestamp_col = col
        if 'lat' in col_lower and 'lon' not in col_lower:
            lat_col = col
        if 'lon' in col_lower or 'lng' in col_lower:
            lon_col = col
        if 'severity' in col_lower:
            severity_col = col
    
    print(f"  Detected columns:")
    print(f"    Timestamp: {timestamp_col}")
    print(f"    Latitude: {lat_col}")
    print(f"    Longitude: {lon_col}")
    print(f"    Severity: {severity_col}")
    
    # Create features
    try:
        df_enhanced = engineer.create_all_features(
            timestamp_column=timestamp_col,
            lat_col=lat_col,
            lon_col=lon_col,
            severity_col=severity_col,
            sensor_columns=['acceleration_x', 'acceleration_y'] if 'acceleration_x' in df.columns else None,
            lag_columns=['estimated_response_time'] if 'estimated_response_time' in df.columns else None,
            grid_size=0.01
        )
        print(f"✓ Feature engineering complete. Shape: {df_enhanced.shape}")
    except Exception as e:
        print(f"  Warning in feature engineering: {e}")
        df_enhanced = df.copy()
    
    # Step 2: Prepare data for modeling
    print("\n[2/5] Preparing data for modeling...")
    
    # Select numeric features
    exclude_cols = ['timestamp', 'severity', 'latitude', 'longitude', 'lat', 'lon']
    feature_cols = [col for col in df_enhanced.columns 
                   if col not in exclude_cols and 
                   df_enhanced[col].dtype in [np.int64, np.float64]]
    
    X = df_enhanced[feature_cols].copy()
    X = X.fillna(X.median())  # Fill missing values
    
    # Prepare target
    if task_type == 'classification' and severity_col and severity_col in df_enhanced.columns:
        le = LabelEncoder()
        y = le.fit_transform(df_enhanced[severity_col])
        class_names = list(le.classes_)
        print(f"  Target classes: {class_names}")
    elif 'estimated_response_time' in df_enhanced.columns:
        y = df_enhanced['estimated_response_time']
        task_type = 'regression'
        class_names = None
        print(f"  Using regression task (response time prediction)")
    else:
        print("  Error: Could not determine target variable")
        return None
    
    print(f"✓ Prepared data. Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    # Step 3: Model Tuning
    print("\n[3/5] Model Tuning...")
    tuner = EnhancedModelTuner(X, y, task_type=task_type, random_state=42)
    tuner.split_data(test_size=0.2)
    
    # Tune models
    cv_folds = 3 if quick_mode else 5
    method = 'random' if quick_mode else 'grid'
    
    print(f"  Using {method} search with {cv_folds}-fold CV...")
    
    try:
        rf_model, rf_params = tuner.tune_random_forest(method=method, cv=cv_folds, n_jobs=-1, verbose=0)
        print(f"  ✓ Random Forest tuned")
    except Exception as e:
        print(f"  Warning in RF tuning: {e}")
        rf_model = None
    
    try:
        gb_model, gb_params = tuner.tune_gradient_boosting(method=method, cv=cv_folds, n_jobs=-1, verbose=0)
        print(f"  ✓ Gradient Boosting tuned")
    except Exception as e:
        print(f"  Warning in GB tuning: {e}")
        gb_model = None
    
    # Select best model
    if rf_model and gb_model:
        best_model = rf_model if tuner.rf_best_score > tuner.gb_best_score else gb_model
        best_name = "Random Forest" if tuner.rf_best_score > tuner.gb_best_score else "Gradient Boosting"
    elif rf_model:
        best_model = rf_model
        best_name = "Random Forest"
    elif gb_model:
        best_model = gb_model
        best_name = "Gradient Boosting"
    else:
        print("  Error: No models trained successfully")
        return None
    
    print(f"  ✓ Best model: {best_name}")
    
    # Step 4: Evaluation
    print("\n[4/5] Model Evaluation...")
    evaluator = ComprehensiveEvaluator(task_type=task_type)
    
    if task_type == 'classification':
        y_pred = best_model.predict(tuner.X_test)
        y_pred_proba = best_model.predict_proba(tuner.X_test) if hasattr(best_model, 'predict_proba') else None
        
        results = evaluator.evaluate_classification(
            tuner.y_test, y_pred, y_pred_proba,
            model_name=best_name,
            class_names=class_names
        )
        
        # Plot confusion matrix
        try:
            evaluator.plot_confusion_matrix(
                tuner.y_test, y_pred,
                model_name=best_name,
                class_names=class_names,
                save_path=str(output_dir / f'confusion_matrix_{best_name.lower().replace(" ", "_")}.png')
            )
        except Exception as e:
            print(f"  Warning: Could not create confusion matrix: {e}")
            
        # Plot ROC curve
        try:
            evaluator.plot_roc_curve(
                tuner.y_test, y_pred_proba,
                model_name=best_name,
                class_names=class_names,
                save_path=str(output_dir / f'roc_curve_{best_name.lower().replace(" ", "_")}.png')
            )
        except Exception as e:
            print(f"  Warning: Could not create ROC curve: {e}")
    else:
        y_pred = best_model.predict(tuner.X_test)
        results = evaluator.evaluate_regression(
            tuner.y_test, y_pred,
            model_name=best_name
        )
        
        # Plot predictions vs actual
        try:
            evaluator.plot_predictions_vs_actual(
                tuner.y_test, y_pred,
                model_name=best_name,
                save_path=str(output_dir / f'predictions_{best_name.lower().replace(" ", "_")}.png')
            )
        except Exception as e:
            print(f"  Warning: Could not create prediction plot: {e}")
    
    # Feature importance
    try:
        evaluator.plot_feature_importance(
            best_model,
            feature_names=list(X.columns),
            top_n=15,
            model_name=best_name,
            save_path=str(output_dir / 'feature_importance.png')
        )
    except Exception as e:
        print(f"  Warning: Could not create feature importance plot: {e}")
    
    # Step 5: Visualizations
    print("\n[5/5] Creating Visualizations...")
    
    try:
        dashboard = VisualizationDashboard(
            df_enhanced,
            lat_col=lat_col,
            lon_col=lon_col,
            timestamp_col=timestamp_col,
            severity_col=severity_col
        )
        
        # Create heatmap
        if lat_col and lon_col:
            try:
                dashboard.create_heatmap_folium(save_path=str(output_dir / 'accident_heatmap.html'))
                print("  ✓ Created heatmap")
            except Exception as e:
                print(f"  Warning: Could not create heatmap: {e}")
        
        # Time-series
        if timestamp_col:
            try:
                dashboard.plot_demand_time_series(frequency='H', save_path=str(output_dir / 'demand_time_series.png'))
                print("  ✓ Created time-series chart")
            except Exception as e:
                print(f"  Warning: Could not create time-series: {e}")
    except Exception as e:
        print(f"  Warning: Could not create visualizations: {e}")
    
    # Save models
    try:
        import joblib
        joblib.dump(best_model, models_dir / f'{best_name.lower().replace(" ", "_")}_model.pkl')
        print(f"  ✓ Saved model to models/{best_name.lower().replace(' ', '_')}_model.pkl")
    except Exception as e:
        print(f"  Warning: Could not save model: {e}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"Generated files in {output_dir}:")
    print("  - Confusion matrix / prediction plots")
    print("  - ROC curve plot")
    print("  - Feature importance plot")
    print("  - Heatmap (if coordinates available)")
    print("  - Time-series chart (if timestamp available)")
    print(f"Models saved in {models_dir}")
    print("\n" + "="*80)
    
    return best_model, X.columns, evaluator


def main():
    """Main function"""
    print("="*80)
    print("ML MODEL RUNNER")
    print("="*80)
    
    # Get data file path from command line or use default
    data_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Load data
    df = load_data(data_file)
    
    # Determine task type
    task_type = 'classification'
    if 'estimated_response_time' in df.columns and 'severity' not in df.columns:
        task_type = 'regression'
    
    # Ask user for quick mode
    quick_mode = True
    if len(sys.argv) > 2:
        quick_mode = sys.argv[2].lower() != 'full'
    
    if quick_mode:
        print("\nRunning in QUICK MODE (faster, less thorough)")
    else:
        print("\nRunning in FULL MODE (slower, more thorough)")
    
    # Run pipeline
    result = run_pipeline(df, task_type=task_type, quick_mode=quick_mode)
    
    if result:
        print("\n✓ Pipeline completed successfully!")
    else:
        print("\n✗ Pipeline encountered errors. Check output above.")


if __name__ == "__main__":
    main()
