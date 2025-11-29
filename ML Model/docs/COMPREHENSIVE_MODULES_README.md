# Comprehensive ML Pipeline Modules

This document describes the four comprehensive modules created for the ML pipeline:

1. **Enhanced Feature Engineering** (`enhanced_feature_engineering.py`)
2. **Enhanced Model Tuning** (`enhanced_model_tuning.py`)
3. **Comprehensive Evaluation** (`comprehensive_evaluation.py`)
4. **Visualization & Integration** (`visualization_integration.py`)

---

## 1. Enhanced Feature Engineering

### Overview
Creates comprehensive features from raw data including temporal, spatial, sensor, frequency-domain, and lag features.

### Key Features

#### Temporal Features
- `day_of_week`, `hour_of_day`, `is_weekend`, `is_rush_hour`
- Cyclical encodings (sin/cos) for periodic features
- Time-of-day categories (morning, afternoon, evening, night)

#### Spatial Grid Features
- Grid-based zones from lat/lon coordinates
- Accident counts and severity statistics per zone
- Distance from grid center

#### Sensor Time-Window Features
- Rolling statistics (mean, std, min, max, median) over 5-second and 10-second windows
- Rate of change features

#### Frequency-Domain Features (FFT)
- Dominant frequency detection
- Spectral energy, centroid, and spread
- Detects sudden changes in sensor patterns

#### Lag Features
- Time-series lag features (1, 2, 3, 5, 10 time steps)
- Difference features (change from previous period)

### Usage

```python
from enhanced_feature_engineering import EnhancedFeatureEngineer

# Initialize
engineer = EnhancedFeatureEngineer(df)

# Create all features
df_enhanced = engineer.create_all_features(
    timestamp_column='timestamp',
    lat_col='latitude',
    lon_col='longitude',
    severity_col='severity',
    sensor_columns=['acceleration_x', 'acceleration_y'],
    lag_columns=['estimated_response_time'],
    grid_size=0.01
)

# Or create features individually
df = engineer.extract_temporal_features('timestamp')
df = engineer.create_spatial_grid('latitude', 'longitude', grid_size=0.01)
df = engineer.create_sensor_rolling_features(['acceleration_x'], window_sizes=[5, 10])
df = engineer.extract_frequency_features(['acceleration_x'])
df = engineer.create_lag_features(['estimated_response_time'], lags=[1, 2, 3])
```

---

## 2. Enhanced Model Tuning

### Overview
Hyperparameter optimization for Random Forest and Gradient Boosting models using GridSearchCV and RandomizedSearchCV.

### Key Features

#### Random Forest Tuning
- Parameters: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- Supports both classification and regression tasks
- 5-fold cross-validation

#### Gradient Boosting Tuning
- Parameters: `learning_rate`, `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `subsample`
- Supports both classification and regression tasks
- 5-fold cross-validation

#### Optimization Metrics
- Classification: F1-score (macro)
- Regression: Mean Squared Error (MSE)

### Usage

```python
from enhanced_model_tuning import EnhancedModelTuner

# Initialize
tuner = EnhancedModelTuner(X, y, task_type='classification', random_state=42)
tuner.split_data(test_size=0.2)

# Tune Random Forest
rf_model, rf_params = tuner.tune_random_forest(method='random', cv=5)

# Tune Gradient Boosting
gb_model, gb_params = tuner.tune_gradient_boosting(method='grid', cv=5)

# Tune both models
results = tuner.tune_all_models(method='random', cv=5)

# Cross-validation
cv_scores = tuner.cross_validate_model(rf_model, cv=5)

# Save models
tuner.save_models(directory='models/')
```

### Parameter Grids

**Random Forest:**
```python
{
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
```

**Gradient Boosting:**
```python
{
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}
```

---

## 3. Comprehensive Evaluation

### Overview
Comprehensive evaluation for both regression and classification models with metrics, visualizations, and comparison tools.

### Key Features

#### Regression Metrics
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)
- Median Absolute Error

#### Classification Metrics
- Accuracy
- Precision (weighted and per-class)
- Recall (weighted and per-class)
- F1-Score (weighted and per-class)
- Confusion Matrix

#### Visualizations
- Confusion matrix heatmaps
- Model comparison charts
- Predictions vs actual plots (regression)
- Feature importance plots
- Cross-validation score distributions

### Usage

```python
from comprehensive_evaluation import ComprehensiveEvaluator

# Initialize
evaluator = ComprehensiveEvaluator(task_type='classification')

# Evaluate classification model
results = evaluator.evaluate_classification(
    y_true, y_pred, y_pred_proba,
    model_name='Random Forest',
    class_names=['Low', 'Medium', 'High']
)

# Evaluate regression model
results = evaluator.evaluate_regression(
    y_true, y_pred,
    model_name='Gradient Boosting'
)

# Plot confusion matrix
evaluator.plot_confusion_matrix(
    y_true, y_pred,
    model_name='Random Forest',
    class_names=['Low', 'Medium', 'High'],
    save_path='confusion_matrix.png'
)

# Compare multiple models
evaluator.compare_models(save_path='model_comparison.png')

# Cross-validation
cv_scores = evaluator.cross_validate(model, X, y, cv=5)

# Feature importance
evaluator.plot_feature_importance(
    model, feature_names,
    top_n=20,
    save_path='feature_importance.png'
)

# Generate summary report
summary = evaluator.generate_summary_report(save_path='evaluation_summary.csv')
```

---

## 4. Visualization & Integration

### Overview
Interactive dashboard system with maps, time-series charts, filtering capabilities, and API structure for real-time predictions.

### Key Features

#### Map Visualizations
- **Folium Heatmaps**: Accident hotspots with heatmap overlay
- **Plotly Maps**: Interactive density maps
- **Ambulance Locations**: Plot ambulance positions and nearby accidents
- **Optimal Routes**: Visualize routes from start to destinations

#### Time-Series Charts
- Demand prediction over time (hourly, daily, weekly)
- Distribution by hour of day or day of week
- Rolling averages

#### Filtering Capabilities
- Filter by date range
- Filter by location (bounding box)
- Filter by severity level

#### API Structure
- Flask-based REST API
- Single prediction endpoint
- Batch prediction endpoint
- Health check endpoint

### Usage

#### Visualization Dashboard

```python
from visualization_integration import VisualizationDashboard

# Initialize
dashboard = VisualizationDashboard(
    df,
    lat_col='latitude',
    lon_col='longitude',
    timestamp_col='timestamp',
    severity_col='severity'
)

# Create heatmap (Folium)
dashboard.create_heatmap_folium(
    severity_filter=['High', 'Critical'],
    date_range=('2024-01-01', '2024-12-31'),
    save_path='accident_heatmap.html'
)

# Create heatmap (Plotly)
dashboard.create_heatmap_plotly(
    severity_filter=['High'],
    save_path='accident_heatmap_plotly.html'
)

# Plot ambulance locations
ambulance_locations = pd.DataFrame({
    'lat': [12.9716, 12.9816],
    'lon': [77.5946, 77.6046],
    'ambulance_id': ['AMB001', 'AMB002']
})
dashboard.plot_ambulance_locations(
    ambulance_locations,
    accident_locations=df,
    save_path='ambulance_map.html'
)

# Plot optimal routes
start = (12.9716, 77.5946)
destinations = [(12.9816, 77.6046), (12.9616, 77.5846)]
dashboard.plot_optimal_routes(
    start, destinations,
    save_path='routes_map.html'
)

# Time-series chart
dashboard.plot_demand_time_series(
    frequency='H',  # 'H' for hourly, 'D' for daily, 'W' for weekly
    severity_filter=['High', 'Medium'],
    save_path='demand_time_series.png'
)

# Interactive dashboard
dashboard.create_interactive_dashboard(save_path='dashboard.html')
```

#### API Setup

```python
from visualization_integration import PredictionAPI

# Initialize API
api = PredictionAPI(
    model=trained_model,
    feature_columns=['feature1', 'feature2', ...],
    scaler=scaler  # Optional
)

# Single prediction (without Flask)
prediction = api.predict_single({
    'feature1': 1.5,
    'feature2': 2.3,
    ...
})

# Run API server
api.run(host='0.0.0.0', port=5000, debug=False)
```

#### API Endpoints

**Health Check:**
```bash
GET http://localhost:5000/health
```

**Single Prediction:**
```bash
POST http://localhost:5000/predict
Content-Type: application/json

{
  "feature1": 1.5,
  "feature2": 2.3,
  ...
}
```

**Batch Prediction:**
```bash
POST http://localhost:5000/predict_batch
Content-Type: application/json

{
  "records": [
    {"feature1": 1.5, "feature2": 2.3, ...},
    {"feature1": 1.8, "feature2": 2.1, ...}
  ]
}
```

---

## Complete Pipeline Example

See `example_comprehensive_pipeline.py` for a complete example that demonstrates all four modules working together.

### Running the Example

```bash
python example_comprehensive_pipeline.py
```

This will:
1. Load or create sample data
2. Perform comprehensive feature engineering
3. Tune Random Forest and Gradient Boosting models
4. Evaluate both models with visualizations
5. Create interactive dashboards and maps
6. Set up API structure

---

## Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 2.2.0
- numpy >= 1.26.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.8.0
- seaborn >= 0.13.0
- scipy >= 1.11.0
- folium >= 0.14.0 (for map visualizations)
- plotly >= 5.17.0 (for interactive visualizations)
- flask >= 2.3.0 (for API)

---

## Output Files

The pipeline generates various output files:

### Evaluation Outputs
- `confusion_matrix_rf.png` - Random Forest confusion matrix
- `confusion_matrix_gb.png` - Gradient Boosting confusion matrix
- `model_comparison.png` - Side-by-side model comparison
- `feature_importance.png` - Top feature importances
- `evaluation_summary.csv` - Summary metrics for all models

### Visualization Outputs
- `accident_heatmap.html` - Interactive heatmap (Folium)
- `accident_heatmap_plotly.html` - Interactive heatmap (Plotly)
- `demand_time_series.png` - Time-series demand chart
- `dashboard.html` - Interactive dashboard
- `ambulance_map.html` - Ambulance locations map
- `routes_map.html` - Optimal routes visualization

### Model Files
- `models/random_forest_model.pkl` - Trained Random Forest model
- `models/gradient_boosting_model.pkl` - Trained Gradient Boosting model

---

## Notes

1. **Feature Engineering**: Some features (like FFT) require sufficient data points. The module handles edge cases gracefully.

2. **Model Tuning**: GridSearchCV can be slow for large datasets. Use RandomizedSearchCV for faster results.

3. **Visualizations**: Folium and Plotly are optional. The modules will work without them but map features won't be available.

4. **API**: The Flask API is optional. You can use the `predict_single()` method without running a server.

5. **Memory**: For very large datasets, consider processing in chunks or using sampling for visualization.

---

## Troubleshooting

**Issue**: `folium not available`
- Solution: `pip install folium`

**Issue**: `plotly not available`
- Solution: `pip install plotly`

**Issue**: `flask not available`
- Solution: `pip install flask`

**Issue**: FFT features showing NaN
- Solution: Ensure sensor data has enough samples (>= n_fft points)

**Issue**: Grid search taking too long
- Solution: Use `method='random'` instead of `method='grid'` or reduce parameter grid size

---

## License

This code is provided as-is for educational and research purposes.

