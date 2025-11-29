# Emergency Response ML Pipeline

A comprehensive machine learning pipeline for emergency response optimization, accident severity prediction, and demand forecasting.

## 📁 Project Structure

```
COE/
├── src/                    # Core source modules
│   ├── feature_engineering/    # Feature engineering modules
│   ├── model_training/          # Model training & tuning
│   ├── evaluation/             # Model evaluation
│   └── visualization/          # Visualization & API
│
├── scripts/               # Executable scripts
│   ├── main.py                 # Original pipeline
│   └── example_comprehensive_pipeline.py  # Complete example
│
├── data/                   # Data files (CSV)
├── docs/                    # Documentation
├── models/                  # Trained models (.pkl)
└── output/                  # Output files (JSON, images)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
python scripts/example_comprehensive_pipeline.py
```

Or from the scripts directory:

```bash
cd scripts
python example_comprehensive_pipeline.py
```

### 3. Run Original Pipeline

```bash
python scripts/main.py
```

## 📚 Documentation

- **[Quick Start Guide](docs/QUICK_START_GUIDE.md)** - Step-by-step instructions
- **[Comprehensive Modules README](docs/COMPREHENSIVE_MODULES_README.md)** - Detailed module documentation
- **[Project Structure](PROJECT_STRUCTURE.md)** - Complete project organization

## 🎯 Features

### 1. Feature Engineering
- **Temporal Features**: day_of_week, hour_of_day, is_weekend, is_rush_hour
- **Spatial Grid**: Grid-based zones with accident aggregation
- **Sensor Features**: Rolling statistics, FFT frequency-domain features
- **Lag Features**: Time-series lag features for prediction

### 2. Model Tuning
- **Random Forest**: Hyperparameter optimization
- **Gradient Boosting**: Hyperparameter optimization
- **Cross-Validation**: 5-fold CV with GridSearchCV/RandomizedSearchCV

### 3. Evaluation
- **Regression Metrics**: MSE, RMSE, MAE, R²
- **Classification Metrics**: Accuracy, Precision, Recall, F1-score
- **Visualizations**: Confusion matrices, model comparisons, feature importance

### 4. Visualization & Integration
- **Interactive Maps**: Folium/Plotly heatmaps
- **Time-Series Charts**: Demand prediction over time
- **Dashboard**: Interactive dashboard with filtering
- **API**: Flask-based REST API for real-time predictions

### 5. Accelerometer Accident Detection
- High-frequency accident classifier built on 100-sample accelerometer windows
- Training entrypoint: `scripts/train_accelerometer_accident_detector.py`
- Detailed design notes: [`docs/ACCELEROMETER_ACCIDENT_DETECTION.md`](docs/ACCELEROMETER_ACCIDENT_DETECTION.md)

## 📖 Usage Examples

### Basic Usage

```python
from src.feature_engineering import EnhancedFeatureEngineer
from src.model_training import EnhancedModelTuner
from src.evaluation import ComprehensiveEvaluator

# Feature engineering
engineer = EnhancedFeatureEngineer(df)
df_enhanced = engineer.create_all_features(
    timestamp_column='timestamp',
    lat_col='latitude',
    lon_col='longitude'
)

# Model tuning
tuner = EnhancedModelTuner(X, y, task_type='classification')
tuner.split_data()
rf_model, _ = tuner.tune_random_forest(method='random', cv=5)

# Evaluation
evaluator = ComprehensiveEvaluator(task_type='classification')
y_pred = rf_model.predict(X_test)
evaluator.evaluate_classification(y_test, y_pred)
```

See [docs/QUICK_START_GUIDE.md](docs/QUICK_START_GUIDE.md) for more examples.

## 📦 Modules

### Enhanced Feature Engineering
- `src/feature_engineering/enhanced_feature_engineering.py`

### Enhanced Model Tuning
- `src/model_training/enhanced_model_tuning.py`

### Comprehensive Evaluation
- `src/evaluation/comprehensive_evaluation.py`

### Visualization & Integration
- `src/visualization/visualization_integration.py`

## 🔧 Requirements

- Python 3.8+
- pandas >= 2.2.0
- numpy >= 1.26.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.8.0
- seaborn >= 0.13.0
- scipy >= 1.11.0
- folium >= 0.14.0 (optional, for maps)
- plotly >= 5.17.0 (optional, for interactive charts)
- flask >= 2.3.0 (optional, for API)

## 📝 Output Files

After running the pipeline, you'll find:

- **Models**: `models/*.pkl`
- **Evaluations**: `*.png` (confusion matrices, comparisons)
- **Visualizations**: `*.html` (maps, dashboards)
- **Reports**: `evaluation_summary.csv`

## 🤝 Contributing

1. Follow the project structure
2. Add modules to appropriate `src/` subdirectories
3. Update documentation
4. Test your changes

## 📄 License

This project is provided as-is for educational and research purposes.

## 🆘 Support

For issues or questions:
1. Check [QUICK_START_GUIDE.md](docs/QUICK_START_GUIDE.md)
2. Review [COMPREHENSIVE_MODULES_README.md](docs/COMPREHENSIVE_MODULES_README.md)
3. See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for organization details
