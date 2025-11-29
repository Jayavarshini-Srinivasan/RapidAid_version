"""
Main Pipeline Script
Orchestrates the complete accident severity prediction pipeline
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_analysis import DataAnalyzer
from src.visualization_output import VisualizationOutput
import warnings
warnings.filterwarnings('ignore')

# Robust import for FeatureEngineer: prefer package, fall back to module file if needed
try:
    from src.feature_engineering import FeatureEngineer
except Exception:
    # If a package named src.feature_engineering exists and doesn't expose FeatureEngineer,
    # fall back to loading the plain module file at src/feature_engineering.py
    import importlib.util
    fe_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'feature_engineering.py')
    if os.path.exists(fe_path):
        spec = importlib.util.spec_from_file_location('feature_engineering_module', fe_path)
        fe_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fe_mod)
        FeatureEngineer = getattr(fe_mod, 'FeatureEngineer')
    else:
        raise

# Robust import for ModelTrainer: prefer package, fall back to module file if needed
try:
    from src.model_training import ModelTrainer
except Exception:
    import importlib.util
    mt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'model_training.py')
    if os.path.exists(mt_path):
        spec = importlib.util.spec_from_file_location('model_training_module', mt_path)
        mt_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mt_mod)
        ModelTrainer = getattr(mt_mod, 'ModelTrainer')
    else:
        raise


def main():
    """Main execution function"""
    
    print("=" * 60)
    print("Accident Severity Prediction Model - XGBoost")
    print("=" * 60)
    
    # Step 1: Data Loading and Analysis
    print("\n[Step 1] Loading and analyzing data...")
    
    # Get data file path
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        # Collect candidate files from current dir and ./data folder
        def list_files_in_dir(path):
            try:
                return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.csv', '.xlsx', '.xls'))]
            except Exception:
                return []
        
        candidates = []
        # top-level
        candidates.extend(list_files_in_dir('.'))
        # data folder
        candidates.extend(list_files_in_dir('data'))
        
        # if no candidates yet, walk recursively
        if len(candidates) == 0:
            for root, _, files in os.walk('.'):
                for f in files:
                    if f.endswith(('.csv', '.xlsx', '.xls')):
                        candidates.append(os.path.join(root, f))
        
        if len(candidates) == 0:
            print("ERROR: No data file found. Please provide a data file path as argument.")
            print("Usage: python main.py <path_to_data_file>")
            return
        
        # Prefer files that match keywords (and prefer the 'data' folder)
        keywords = ['india', 'traffic', 'accident', 'accidents', 'accients', 'dispatch', 'synthetic']
        def score_file(fp):
            fn = os.path.basename(fp).lower()
            score = 0
            for k in keywords:
                if k in fn:
                    score += 1
            if os.path.dirname(fp).endswith('data'):
                score += 1  # prefer files inside data/
            return score
        
        candidates.sort(key=lambda p: (score_file(p), os.path.basename(p)), reverse=True)
        data_file = candidates[0]
        print(f"Using data file: {data_file}")
    
    # Initialize data analyzer
    analyzer = DataAnalyzer(data_file)
    df = analyzer.load_data()
    
    # Analyze missing values
    missing_info = analyzer.analyze_missing_values()
    
    # Get data summary
    analyzer.get_data_summary()
    
    # Convert dates and combine with time
    df = analyzer.convert_dates(date_columns=['date'], time_column='time')
    
    # Normalize severity (map High to Critical)
    df = analyzer.normalize_severity(severity_column='severity')
    
    # Handle missing values
    df = analyzer.handle_missing_values(strategy='auto')
    
    # Step 2: Feature Engineering
    print("\n[Step 2] Feature engineering...")
    
    feature_engineer = FeatureEngineer(df)
    
    # Identify column types
    categorical_cols, numerical_cols = feature_engineer.identify_column_types()
    
    # Extract temporal features (use datetime column if created, otherwise date)
    if 'datetime' in df.columns:
        df = feature_engineer.extract_temporal_features(date_column='datetime')
    else:
        df = feature_engineer.extract_temporal_features(date_column='date')
    
    # Process geospatial data
    df = feature_engineer.process_geospatial_data()
    
    # Encode categorical variables
    df = feature_engineer.encode_categorical_variables(encoding_type='label')
    
    # Scale numerical features (optional - Random Forest doesn't require scaling)
    # df = feature_engineer.scale_numerical_features()
    
    # Prepare features
    # Find target column (case-insensitive)
    target_col = None
    for col in df.columns:
        if 'severity' in col.lower():
            target_col = col
            break
    
    if target_col is None:
        print("WARNING: 'severity' column not found. Please ensure your data has a severity column.")
        print("Available columns:", list(df.columns))
        return
    
    X, y, feature_columns = feature_engineer.prepare_features(
        target_column=target_col,
        exclude_columns=['latitude', 'longitude', 'lat', 'lon', 'lng', 
                        'Latitude', 'Longitude', 'LAT', 'LON', 'date', 'time', 'datetime']
    )
    
    # Store target encoder for later use
    target_encoder = feature_engineer.target_encoder
    
    # Update df to include all engineered features
    df = feature_engineer.df
    
    # Step 3: Model Training
    print("\n[Step 3] Training Random Forest model...")
    
    trainer = ModelTrainer(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(test_size=0.2, random_state=42)
    
    # Train baseline model (XGBoost)
    baseline_model = trainer.train_xgboost_model()
    
    # Hyperparameter tuning — use randomized search (faster for large datasets)
    best_model, best_params = trainer.randomized_hyperparameter_tuning_xgboost(n_iter=30, cv=3, n_jobs=-1)
    
    # Cross-validation
    cv_scores = trainer.cross_validate(model=best_model, cv=5)
    
    # Evaluate model
    evaluation_results = trainer.evaluate_model(model=best_model)
    
    # Save model
    model_path = trainer.save_model('models/accident_severity_model.pkl')
    
    # Step 4: Generate Output for Visualization
    print("\n[Step 4] Generating visualization output...")
    
    # Get predictions for test set
    predictions = best_model.predict(X_test)
    
    # Create visualization output
    viz_output = VisualizationOutput(
        df=df,
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        predictions=predictions,
        feature_columns=feature_columns,
        target_encoder=target_encoder
    )
    
    # Generate predictions JSON
    predictions_json = viz_output.generate_predictions_json('output/predictions.json')
    
    # Generate summary statistics
    summary_json = viz_output.generate_summary_statistics('output/summary_stats.json')

    # Generate ROC Curve
    try:
        from src.evaluation.comprehensive_evaluation import ComprehensiveEvaluator
        evaluator = ComprehensiveEvaluator(task_type='classification')
        y_pred_proba = best_model.predict_proba(X_test)
        
        # Get class names
        class_names = None
        if target_encoder is not None:
            class_names = list(target_encoder.classes_)
        
        evaluator.plot_roc_curve(
            y_test, y_pred_proba,
            model_name='XGBoost',
            class_names=class_names,
            save_path='output/roc_curve.png'
        )
        print("  ROC Curve: output/roc_curve.png")
    except Exception as e:
        print(f"  Warning: Could not generate ROC curve: {e}")
    
    # Step 5: Print Final Summary
    print("\n" + "=" * 60)
    print("Pipeline Execution Complete!")
    print("=" * 60)
    print(f"\nModel Performance:")
    print(f"  Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"  Precision: {evaluation_results['precision']:.4f}")
    print(f"  Recall: {evaluation_results['recall']:.4f}")
    print(f"  F1-Score: {evaluation_results['f1_score']:.4f}")
    print(f"\nOutput Files:")
    print(f"  Model: {model_path}")
    print(f"  Predictions JSON: {predictions_json}")
    print(f"  Summary Statistics: {summary_json}")
    print(f"  ROC Curve: output/roc_curve.png")
    print(f"\nTop 5 Most Important Features:")
    for idx, row in evaluation_results['feature_importance'].head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

