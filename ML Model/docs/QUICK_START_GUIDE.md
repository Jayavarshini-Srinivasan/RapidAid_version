# Quick Start Guide - Running the ML Models

This guide will help you run the comprehensive ML pipeline modules.

## Step 1: Install Dependencies

First, make sure all required packages are installed:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib folium plotly flask
```

## Step 2: Prepare Your Data

Your data should have these columns (or similar):
- **Timestamp**: `timestamp`, `datetime`, or `date`
- **Location**: `latitude`/`longitude` (or `lat`/`lon`)
- **Severity**: `severity` (for classification tasks)
- **Sensor data**: `acceleration_x`, `acceleration_y`, etc. (optional)
- **Target variable**: For regression (e.g., `estimated_response_time`)

## Step 3: Run the Complete Pipeline

### Option A: Run the Example Pipeline (Recommended for First Time)

```bash
python example_comprehensive_pipeline.py
```

This will:
1. Load your data (or create sample data if files don't exist)
2. Perform feature engineering
3. Tune Random Forest and Gradient Boosting models
4. Evaluate both models
5. Generate visualizations
6. Set up API structure

### Option B: Use Individual Modules

Create your own script:

```python
import pandas as pd
from enhanced_feature_engineering import EnhancedFeatureEngineer
from enhanced_model_tuning import EnhancedModelTuner
from comprehensive_evaluation import ComprehensiveEvaluator
from visualization_integration import VisualizationDashboard, PredictionAPI

# 1. Load your data
df = pd.read_csv('your_data.csv')

# 2. Feature Engineering
engineer = EnhancedFeatureEngineer(df)
df_enhanced = engineer.create_all_features(
    timestamp_column='timestamp',
    lat_col='latitude',
    lon_col='longitude',
    severity_col='severity'
)

# 3. Prepare features for modeling
# Select numeric features
feature_cols = [col for col in df_enhanced.columns 
                if col not in ['timestamp', 'severity', 'latitude', 'longitude']]
X = df_enhanced[feature_cols].select_dtypes(include=['number'])
X = X.fillna(X.median())  # Handle missing values

# Prepare target
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df_enhanced['severity'])

# 4. Model Tuning
tuner = EnhancedModelTuner(X, y, task_type='classification')
tuner.split_data(test_size=0.2)

# Tune models (use 'random' for faster results)
rf_model, rf_params = tuner.tune_random_forest(method='random', cv=5)
gb_model, gb_params = tuner.tune_gradient_boosting(method='random', cv=5)

# 5. Evaluation
evaluator = ComprehensiveEvaluator(task_type='classification')
y_pred = rf_model.predict(tuner.X_test)
results = evaluator.evaluate_classification(
    tuner.y_test, y_pred, 
    model_name='Random Forest',
    class_names=list(le.classes_)
)

# 6. Visualizations
dashboard = VisualizationDashboard(df_enhanced)
dashboard.create_heatmap_folium(save_path='heatmap.html')
dashboard.plot_demand_time_series(save_path='demand.png')
```

## Step 4: Common Use Cases

### Use Case 1: Classification (Accident Severity Prediction)

```python
import pandas as pd
from enhanced_feature_engineering import EnhancedFeatureEngineer
from enhanced_model_tuning import EnhancedModelTuner
from comprehensive_evaluation import ComprehensiveEvaluator
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('synthetic_dispatch_data.csv')

# Feature engineering
engineer = EnhancedFeatureEngineer(df)
df_enhanced = engineer.create_all_features(
    timestamp_column='timestamp',
    lat_col='latitude',
    lon_col='longitude',
    severity_col='severity'
)

# Prepare data
feature_cols = [col for col in df_enhanced.columns 
                if col not in ['timestamp', 'severity', 'latitude', 'longitude']]
X = df_enhanced[feature_cols].select_dtypes(include=['number']).fillna(0)
y = LabelEncoder().fit_transform(df_enhanced['severity'])

# Train and tune
tuner = EnhancedModelTuner(X, y, task_type='classification')
tuner.split_data()
rf_model, _ = tuner.tune_random_forest(method='random', cv=3)

# Evaluate
evaluator = ComprehensiveEvaluator(task_type='classification')
y_pred = rf_model.predict(tuner.X_test)
evaluator.evaluate_classification(tuner.y_test, y_pred, model_name='RF')
```

### Use Case 2: Regression (Response Time Prediction)

```python
import pandas as pd
from enhanced_feature_engineering import EnhancedFeatureEngineer
from enhanced_model_tuning import EnhancedModelTuner
from comprehensive_evaluation import ComprehensiveEvaluator

# Load data
df = pd.read_csv('synthetic_dispatch_data.csv')

# Feature engineering
engineer = EnhancedFeatureEngineer(df)
df_enhanced = engineer.create_all_features(
    timestamp_column='timestamp',
    lat_col='latitude',
    lon_col='longitude'
)

# Prepare data
feature_cols = [col for col in df_enhanced.columns 
                if col not in ['timestamp', 'estimated_response_time', 'latitude', 'longitude']]
X = df_enhanced[feature_cols].select_dtypes(include=['number']).fillna(0)
y = df_enhanced['estimated_response_time']

# Train and tune
tuner = EnhancedModelTuner(X, y, task_type='regression')
tuner.split_data()
gb_model, _ = tuner.tune_gradient_boosting(method='random', cv=3)

# Evaluate
evaluator = ComprehensiveEvaluator(task_type='regression')
y_pred = gb_model.predict(tuner.X_test)
evaluator.evaluate_regression(tuner.y_test, y_pred, model_name='GB')
evaluator.plot_predictions_vs_actual(tuner.y_test, y_pred, model_name='GB')
```

### Use Case 3: Quick Model Training (No Tuning)

If you want to train models quickly without hyperparameter tuning:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from comprehensive_evaluation import ComprehensiveEvaluator

# Prepare data (after feature engineering)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
evaluator = ComprehensiveEvaluator(task_type='classification')
y_pred = model.predict(X_test)
evaluator.evaluate_classification(y_test, y_pred, model_name='RF Baseline')
```

## Step 5: Generate Visualizations

### Create Maps and Dashboards

```python
from visualization_integration import VisualizationDashboard

# Initialize dashboard
dashboard = VisualizationDashboard(
    df,
    lat_col='latitude',
    lon_col='longitude',
    timestamp_col='timestamp',
    severity_col='severity'
)

# Create heatmap
dashboard.create_heatmap_folium(
    severity_filter=['High', 'Critical'],
    save_path='accident_heatmap.html'
)

# Time-series chart
dashboard.plot_demand_time_series(
    frequency='H',  # Hourly
    save_path='demand_time_series.png'
)

# Interactive dashboard
dashboard.create_interactive_dashboard(save_path='dashboard.html')
```

## Step 6: Set Up API for Real-Time Predictions

### Option A: Use API Without Server (Single Predictions)

```python
from visualization_integration import PredictionAPI

# Initialize API
api = PredictionAPI(
    model=trained_model,
    feature_columns=list(X.columns),
    scaler=None
)

# Make prediction
sample_features = {
    'feature1': 1.5,
    'feature2': 2.3,
    # ... all required features
}
prediction = api.predict_single(sample_features)
print(prediction)
```

### Option B: Run API Server

```python
from visualization_integration import PredictionAPI

# Initialize API
api = PredictionAPI(
    model=trained_model,
    feature_columns=list(X.columns)
)

# Start server
api.run(host='0.0.0.0', port=5000, debug=False)
```

Then make API calls:
```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 1.5, "feature2": 2.3, ...}'

# Batch prediction
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"records": [{"feature1": 1.5, ...}, {"feature1": 1.8, ...}]}'
```

## Troubleshooting

### Issue: "No module named 'folium'"
```bash
pip install folium
```

### Issue: "No module named 'plotly'"
```bash
pip install plotly
```

### Issue: Data has missing values
```python
# Fill missing values before feature engineering
df = df.fillna(df.median())  # For numeric columns
df = df.fillna(df.mode().iloc[0])  # For categorical columns
```

### Issue: Model tuning takes too long
```python
# Use randomized search instead of grid search
tuner.tune_random_forest(method='random', cv=3)  # Instead of cv=5
```

### Issue: Out of memory
```python
# Work with a sample of data
df_sample = df.sample(n=10000, random_state=42)
```

## Expected Output Files

After running the pipeline, you should see:

**Evaluation Files:**
- `confusion_matrix_rf.png` - Confusion matrix for Random Forest
- `confusion_matrix_gb.png` - Confusion matrix for Gradient Boosting
- `model_comparison.png` - Side-by-side model comparison
- `feature_importance.png` - Top feature importances
- `evaluation_summary.csv` - Summary metrics

**Visualization Files:**
- `accident_heatmap.html` - Interactive heatmap
- `demand_time_series.png` - Time-series chart
- `dashboard.html` - Interactive dashboard

**Model Files:**
- `models/random_forest_model.pkl` - Trained Random Forest
- `models/gradient_boosting_model.pkl` - Trained Gradient Boosting

## Next Steps

1. **Experiment with different parameters**: Adjust grid sizes, window sizes, etc.
2. **Try different models**: Add XGBoost, LightGBM, etc.
3. **Feature selection**: Use feature importance to select top features
4. **Deploy model**: Use the API structure for production deployment

## Need Help?

Check the detailed documentation in `COMPREHENSIVE_MODULES_README.md` for more information about each module.


