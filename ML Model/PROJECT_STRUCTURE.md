# Project Structure

This document describes the organized structure of the ML pipeline project.

## Directory Structure

```
COE/
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_analysis.py          # Data loading and preprocessing
│   ├── feature_engineering.py    # Original feature engineering
│   ├── model_training.py         # Original model training
│   ├── visualization_output.py   # Original visualization
│   │
│   ├── feature_engineering/      # Enhanced feature engineering module
│   │   ├── __init__.py
│   │   ├── enhanced_feature_engineering.py
│   │   └── accelerometer_feature_engineer.py
│   │
│   ├── model_training/           # Enhanced model tuning module
│   │   ├── __init__.py
│   │   ├── enhanced_model_tuning.py
│   │   └── accelerometer_accident_detector.py
│   │
│   ├── evaluation/               # Model evaluation module
│   │   ├── __init__.py
│   │   └── comprehensive_evaluation.py
│   │
│   └── visualization/            # Visualization & integration module
│       ├── __init__.py
│       └── visualization_integration.py
│
├── scripts/                      # Executable scripts
│   ├── main.py                   # Main pipeline script (original)
│   ├── example_comprehensive_pipeline.py  # Complete example pipeline
│   ├── run_models.py             # Model running script
│   ├── generate_sample_data.py  # Sample data generator
│   ├── EMS_Data.py              # EMS data processing
│   ├── Synthetic_Data.py        # Synthetic data generation
│   └── train_accelerometer_accident_detector.py  # Accelerometer accident model
│
├── data/                         # Data files
│   ├── india_traffic_accidents.csv
│   ├── synthetic_dispatch_data.csv
│   └── synthetic_sensor_data.csv
│
├── docs/                         # Documentation
│   ├── README.md                 # Main README (in root)
│   ├── COMPREHENSIVE_MODULES_README.md  # Detailed module docs
│   ├── QUICK_START_GUIDE.md      # Quick start guide
│   ├── QUICKSTART.md             # Quick start (original)
│   └── ACCELEROMETER_ACCIDENT_DETECTION.md  # Accelerometer pipeline guide
│
├── models/                       # Trained models (saved as .pkl files)
│   └── accident_severity_model.pkl
│
├── output/                       # Output files
│   ├── predictions.json
│   └── summary_stats.json
│
├── requirements.txt              # Python dependencies
└── PROJECT_STRUCTURE.md          # This file
```

## Module Organization

### Core Modules (`src/`)

#### Feature Engineering
- **`src/feature_engineering.py`**: Original feature engineering module
- **`src/feature_engineering/enhanced_feature_engineering.py`**: Enhanced version with:
  - Temporal features (day_of_week, hour_of_day, is_weekend, is_rush_hour)
  - Spatial grid-based zones
  - Sensor rolling statistics
  - Frequency-domain features (FFT)
  - Lag features
- **`src/feature_engineering/accelerometer_feature_engineer.py`**: Windowed accelerometer feature generator with statistical, jerk, energy, correlation, and FFT descriptors.

#### Model Training
- **`src/model_training.py`**: Original model training module
- **`src/model_training/enhanced_model_tuning.py`**: Enhanced version with:
  - Random Forest hyperparameter tuning
  - Gradient Boosting hyperparameter tuning
  - GridSearchCV and RandomizedSearchCV
  - Cross-validation support
- **`src/model_training/accelerometer_accident_detector.py`**: XGBoost-based accident classifier with threshold optimisation and export helpers.

#### Evaluation
- **`src/evaluation/comprehensive_evaluation.py`**: Comprehensive evaluation with:
  - Regression metrics (MSE, RMSE, MAE, R²)
  - Classification metrics (Accuracy, Precision, Recall, F1)
  - Confusion matrix visualizations
  - Model comparison charts
  - Feature importance plots

#### Visualization
- **`src/visualization_output.py`**: Original visualization module
- **`src/visualization/visualization_integration.py`**: Enhanced version with:
  - Interactive heatmaps (Folium/Plotly)
  - Ambulance location mapping
  - Time-series charts
  - Dashboard creation
  - API structure for real-time predictions

### Scripts (`scripts/`)

- **`main.py`**: Original main pipeline script
- **`example_comprehensive_pipeline.py`**: Complete example demonstrating all modules
- **`run_models.py`**: Script for running models
- **`generate_sample_data.py`**: Generate sample data for testing
- **`EMS_Data.py`**: EMS data processing
- **`Synthetic_Data.py`**: Synthetic data generation
- **`train_accelerometer_accident_detector.py`**: CLI to train/evaluate the accelerometer model.

### Data (`data/`)

All CSV data files are stored here:
- Traffic accident data
- Synthetic dispatch data
- Sensor data

### Documentation (`docs/`)

- **`COMPREHENSIVE_MODULES_README.md`**: Detailed documentation for all modules
- **`QUICK_START_GUIDE.md`**: Step-by-step guide to run the models
- **`QUICKSTART.md`**: Original quick start guide
- **`ACCELEROMETER_ACCIDENT_DETECTION.md`**: Design notes for the accelerometer accident pipeline.

## Import Paths

### From Scripts

When importing from scripts, use:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import EnhancedFeatureEngineer
from src.model_training import EnhancedModelTuner
from src.evaluation import ComprehensiveEvaluator
from src.visualization import VisualizationDashboard, PredictionAPI
```

### From Root Directory

When running from the root directory:

```python
from src.feature_engineering import EnhancedFeatureEngineer
from src.model_training import EnhancedModelTuner
from src.evaluation import ComprehensiveEvaluator
from src.visualization import VisualizationDashboard, PredictionAPI
```

## Running the Pipeline

### Option 1: Run from Scripts Directory

```bash
cd scripts
python example_comprehensive_pipeline.py
```

### Option 2: Run from Root Directory

```bash
python scripts/example_comprehensive_pipeline.py
```

### Option 3: Run Original Pipeline

```bash
python scripts/main.py
```

## File Naming Conventions

- **Modules**: snake_case (e.g., `enhanced_feature_engineering.py`)
- **Classes**: PascalCase (e.g., `EnhancedFeatureEngineer`)
- **Functions**: snake_case (e.g., `extract_temporal_features`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_ITERATIONS`)

## Adding New Modules

1. **Core modules**: Add to appropriate subdirectory in `src/`
2. **Scripts**: Add to `scripts/`
3. **Documentation**: Add to `docs/`
4. **Data**: Add to `data/`

Remember to:
- Create `__init__.py` files for new packages
- Update import paths in existing scripts
- Add documentation

## Best Practices

1. **Keep modules focused**: Each module should have a single responsibility
2. **Use relative imports**: When possible, use relative imports within packages
3. **Document functions**: Add docstrings to all public functions
4. **Handle errors**: Include proper error handling and user-friendly messages
5. **Version control**: Keep data files out of version control (use .gitignore)

## Dependencies

All dependencies are listed in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

