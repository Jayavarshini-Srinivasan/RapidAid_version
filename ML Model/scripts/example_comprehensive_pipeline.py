"""
Comprehensive Example Pipeline
Demonstrates all 4 tasks:
1. Feature Engineering
2. Model Tuning
3. Evaluation
4. Visualization & Integration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import EnhancedFeatureEngineer
from src.model_training import EnhancedModelTuner
from src.evaluation import ComprehensiveEvaluator
from src.visualization import VisualizationDashboard, PredictionAPI

print("="*80)
print("COMPREHENSIVE ML PIPELINE EXAMPLE")
print("="*80)


# ==================== STEP 1: LOAD DATA ====================
print("\n[STEP 1] Loading data...")

# Try to load existing data files
try:
    # Try dispatch data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'synthetic_dispatch_data.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, nrows=10000)  # Limit for demo
    else:
        df = None
    print(f"✓ Loaded {len(df)} records from synthetic_dispatch_data.csv")
except Exception as e:
    # Create sample data if files don't exist
    print(f"Could not load data file: {e}")
    print("Creating sample data...")
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='1H')
    df = pd.DataFrame({
        'timestamp': dates,
        'latitude': np.random.normal(12.9716, 0.1, n_samples),  # Bangalore coordinates
        'longitude': np.random.normal(77.5946, 0.1, n_samples),
        'severity': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.5, 0.3, 0.2]),
        'acceleration_x': np.random.normal(0, 1, n_samples),
        'acceleration_y': np.random.normal(0, 1, n_samples),
        'acceleration_z': np.random.normal(0, 1, n_samples),
        'estimated_response_time': np.random.normal(10, 3, n_samples)
    })
    print(f"✓ Created {len(df)} sample records")

print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")


# ==================== STEP 2: FEATURE ENGINEERING ====================
print("\n" + "="*80)
print("[STEP 2] FEATURE ENGINEERING")
print("="*80)

engineer = EnhancedFeatureEngineer(df)

# Create all features
df_enhanced = engineer.create_all_features(
    timestamp_column='timestamp',
    lat_col='latitude',
    lon_col='longitude',
    severity_col='severity',
    sensor_columns=['acceleration_x', 'acceleration_y', 'acceleration_z'],
    lag_columns=['estimated_response_time'],
    grid_size=0.01
)

print(f"\nEnhanced data shape: {df_enhanced.shape}")
print(f"New features created: {df_enhanced.shape[1] - df.shape[1]}")


# ==================== STEP 3: PREPARE DATA FOR MODELING ====================
print("\n" + "="*80)
print("[STEP 3] PREPARING DATA FOR MODELING")
print("="*80)

# Select features (exclude original columns that were used for feature engineering)
feature_cols = [col for col in df_enhanced.columns 
                if col not in ['timestamp', 'severity', 'latitude', 'longitude'] 
                and not col.startswith('acceleration_')]  # Keep engineered versions

# Remove columns with too many NaN values
feature_cols = [col for col in feature_cols 
                if df_enhanced[col].notna().sum() > len(df_enhanced) * 0.5]

# Fill remaining NaN values
df_enhanced[feature_cols] = df_enhanced[feature_cols].fillna(df_enhanced[feature_cols].median())

# Prepare X and y
X = df_enhanced[feature_cols].select_dtypes(include=[np.number])
y = df_enhanced['severity']

# Encode target if needed
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Features shape: {X.shape}")
print(f"Target classes: {le.classes_}")


# ==================== STEP 4: MODEL TUNING ====================
print("\n" + "="*80)
print("[STEP 4] MODEL TUNING")
print("="*80)

tuner = EnhancedModelTuner(X, y_encoded, task_type='classification', random_state=42)
tuner.split_data(test_size=0.2)

# Tune both models (using random search for faster execution)
print("\nTuning Random Forest...")
rf_model, rf_params = tuner.tune_random_forest(method='random', cv=3, n_jobs=-1)

print("\nTuning Gradient Boosting...")
gb_model, gb_params = tuner.tune_gradient_boosting(method='random', cv=3, n_jobs=-1)

# Get best model
if tuner.rf_best_score > tuner.gb_best_score:
    best_model = rf_model
    best_model_name = "Random Forest"
else:
    best_model = gb_model
    best_model_name = "Gradient Boosting"

print(f"\n✓ Best model: {best_model_name}")


# ==================== STEP 5: MODEL EVALUATION ====================
print("\n" + "="*80)
print("[STEP 5] MODEL EVALUATION")
print("="*80)

evaluator = ComprehensiveEvaluator(task_type='classification')

# Evaluate both models
y_pred_rf = rf_model.predict(tuner.X_test)
y_pred_proba_rf = rf_model.predict_proba(tuner.X_test) if hasattr(rf_model, 'predict_proba') else None

y_pred_gb = gb_model.predict(tuner.X_test)
y_pred_proba_gb = gb_model.predict_proba(tuner.X_test) if hasattr(gb_model, 'predict_proba') else None

# Evaluate Random Forest
print("\nEvaluating Random Forest...")
rf_results = evaluator.evaluate_classification(
    tuner.y_test, y_pred_rf, y_pred_proba_rf, 
    model_name='Random Forest', 
    class_names=list(le.classes_)
)

# Evaluate Gradient Boosting
print("\nEvaluating Gradient Boosting...")
gb_results = evaluator.evaluate_classification(
    tuner.y_test, y_pred_gb, y_pred_proba_gb,
    model_name='Gradient Boosting',
    class_names=list(le.classes_)
)

# Plot confusion matrices
print("\nGenerating confusion matrices...")
evaluator.plot_confusion_matrix(
    tuner.y_test, y_pred_rf, 
    model_name='Random Forest',
    class_names=list(le.classes_),
    save_path='confusion_matrix_rf.png'
)

evaluator.plot_confusion_matrix(
    tuner.y_test, y_pred_gb,
    model_name='Gradient Boosting',
    class_names=list(le.classes_),
    save_path='confusion_matrix_gb.png'
)

# Plot ROC curves
print("\nGenerating ROC curves...")
evaluator.plot_roc_curve(
    tuner.y_test, y_pred_proba_rf,
    model_name='Random Forest',
    class_names=list(le.classes_),
    save_path='roc_curve_rf.png'
)

evaluator.plot_roc_curve(
    tuner.y_test, y_pred_proba_gb,
    model_name='Gradient Boosting',
    class_names=list(le.classes_),
    save_path='roc_curve_gb.png'
)

# Compare models
print("\nGenerating model comparison...")
evaluator.compare_models(save_path='model_comparison.png')

# Cross-validation
print("\nPerforming cross-validation...")
cv_scores = evaluator.cross_validate(best_model, tuner.X_train, tuner.y_train, cv=5)

# Feature importance
print("\nPlotting feature importance...")
evaluator.plot_feature_importance(
    best_model, 
    feature_names=list(X.columns),
    top_n=15,
    model_name=best_model_name,
    save_path='feature_importance.png'
)

# Generate summary report
summary = evaluator.generate_summary_report(save_path='evaluation_summary.csv')
print(f"\n✓ Summary report saved to evaluation_summary.csv")


# ==================== STEP 6: VISUALIZATION ====================
print("\n" + "="*80)
print("[STEP 6] VISUALIZATION & DASHBOARD")
print("="*80)

dashboard = VisualizationDashboard(
    df_enhanced,
    lat_col='latitude',
    lon_col='longitude',
    timestamp_col='timestamp',
    severity_col='severity'
)

# Create heatmap
print("\nCreating heatmap...")
try:
    dashboard.create_heatmap_folium(
        severity_filter=['High', 'Medium'],
        save_path='accident_heatmap.html'
    )
except Exception as e:
    print(f"Could not create folium heatmap: {e}")

# Create time-series plot
print("\nCreating time-series plot...")
try:
    dashboard.plot_demand_time_series(
        frequency='H',
        save_path='demand_time_series.png'
    )
except Exception as e:
    print(f"Could not create time-series plot: {e}")

# Create interactive dashboard
print("\nCreating interactive dashboard...")
try:
    dashboard.create_interactive_dashboard(save_path='dashboard.html')
except Exception as e:
    print(f"Could not create interactive dashboard: {e}")


# ==================== STEP 7: API SETUP ====================
print("\n" + "="*80)
print("[STEP 7] API SETUP")
print("="*80)

# Create API instance
api = PredictionAPI(
    model=best_model,
    feature_columns=list(X.columns),
    scaler=None  # Add scaler if features were scaled
)

# Example: Single prediction
print("\nExample: Single prediction")
sample_features = X.iloc[0].to_dict()
prediction = api.predict_single(sample_features)
print(f"Sample prediction: {prediction}")

# API server (commented out - uncomment to run)
# print("\nStarting API server...")
# print("API will be available at http://localhost:5000")
# print("Endpoints:")
# print("  GET  /health - Health check")
# print("  POST /predict - Single prediction")
# print("  POST /predict_batch - Batch predictions")
# api.run(host='0.0.0.0', port=5000, debug=False)


# ==================== SUMMARY ====================
print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  - confusion_matrix_rf.png")
print("  - confusion_matrix_gb.png")
print("  - roc_curve_rf.png")
print("  - roc_curve_gb.png")
print("  - model_comparison.png")
print("  - feature_importance.png")
print("  - evaluation_summary.csv")
print("  - accident_heatmap.html")
print("  - demand_time_series.png")
print("  - dashboard.html")
print("\n" + "="*80)

