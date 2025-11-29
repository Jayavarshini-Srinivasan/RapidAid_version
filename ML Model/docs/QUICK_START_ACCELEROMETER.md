# Quick Start: Accelerometer Accident Detection

## Step 1: Install Dependencies

```bash
cd /Users/mtejeshx37/COE
pip install -r requirements.txt
```

**Note**: Make sure `xgboost>=2.1.1` is installed (it's in requirements.txt).

## Step 2: Prepare Your Data

Your CSV file should have these columns:
- `accel_x`, `accel_y`, `accel_z` - accelerometer readings (m/s²)
- `accident` - binary label (0 = normal, 1 = accident)
- `timestamp` - timestamp (optional, will be auto-generated if missing)
- `vehicle_id` - vehicle identifier (optional, defaults to "vehicle_0")
- `event_id` - event identifier (optional)
- `event_severity` - severity score (optional)

**Example data format:**
```csv
accel_x,accel_y,accel_z,accident,event_id
-0.33,-1.02,9.59,True,0
0.38,-0.18,10.16,True,0
0.16,0.68,9.78,True,0
```

## Step 3: Train the Model

### Basic Usage (with defaults)
```bash
python scripts/train_accelerometer_accident_detector.py \
  --data-path scripts/synthetic_accident_sensor_data.csv
```

### Full Options
```bash
python scripts/train_accelerometer_accident_detector.py \
  --data-path scripts/synthetic_accident_sensor_data.csv \
  --sampling-rate 50 \
  --window-size 100 \
  --step-size 50 \
  --label-col accident \
  --timestamp-col timestamp \
  --vehicle-id-col vehicle_id \
  --model-path models/accelerometer_accident_detector.pkl \
  --metrics-path output/accelerometer_metrics.json
```

### What Happens:
1. Loads accelerometer data from CSV
2. Windows data into 100-sample segments (with 50% overlap by default)
3. Engineers features (statistical, jerk, energy, FFT, etc.)
4. Trains XGBoost classifier with auto-tuned class weights
5. Evaluates on test set and optimizes decision threshold
6. Saves model and metrics

### Output Files:
- **Model**: `models/accelerometer_accident_detector.pkl`
- **Metrics**: `output/accelerometer_metrics.json`
- **Features** (optional): `output/accelerometer_features.parquet`

## Step 4: Check Results

After training, you'll see output like:
```
[+] Training complete
    Accuracy : 0.950
    Precision: 0.920
    Recall   : 0.890
    ROC-AUC  : 0.980
    Optimal threshold for deployment: 0.450
```

View detailed metrics:
```bash
cat output/accelerometer_metrics.json
```

## Step 5: Use the Model for Predictions

See `scripts/predict_accidents.py` for inference examples, or use the Python API:

```python
import joblib
import pandas as pd
import numpy as np
from src.feature_engineering import AccelerometerFeatureEngineer

# Load trained model
pipeline = joblib.load('models/accelerometer_accident_detector.pkl')

# Load your new accelerometer data
df = pd.read_csv('new_sensor_data.csv')

# Engineer features (same as training)
engineer = AccelerometerFeatureEngineer(
    sampling_rate=50,
    window_size=100,
    step_size=50
)
features_df = engineer.transform(
    df=df,
    sensor_cols=('accel_x', 'accel_y', 'accel_z'),
    label_col='accident'  # Can be dummy column if unlabeled
)

# Get predictions
feature_cols = [col for col in features_df.columns 
                if col not in ['label', 'vehicle_id', 'window_index', 
                              'window_start_ts', 'window_end_ts']]
X = features_df[feature_cols].values
probabilities = pipeline.predict_proba(X)[:, 1]
predictions = (probabilities >= 0.45).astype(int)  # Use optimal threshold

# Add predictions to dataframe
features_df['accident_probability'] = probabilities
features_df['predicted_accident'] = predictions
```

## Troubleshooting

### Error: "Missing sensor columns"
- Ensure your CSV has `accel_x`, `accel_y`, `accel_z` columns
- Use `--label-col` to specify a different label column name

### Error: "No windows were generated"
- Your data might be too short (need at least `window_size` samples)
- Check that `vehicle_id` groups have enough samples

### Low Performance
- Ensure balanced dataset (both normal and accident samples)
- Try adjusting `--window-size` (larger = more context, smaller = faster)
- Check data quality (no extreme outliers, proper sensor calibration)

### Memory Issues
- For very large datasets, consider sampling or chunking the CSV
- Reduce `--window-size` or `--step-size` to generate fewer windows

## Next Steps

1. **Fine-tune hyperparameters**: Edit `AccelerometerAccidentDetector.model_params`
2. **Deploy to production**: Use the saved pipeline for real-time inference
3. **Monitor performance**: Track predictions and update model periodically
4. **Extend features**: Add gyroscope, GPS speed, or other sensor data

For detailed documentation, see `docs/ACCELEROMETER_ACCIDENT_DETECTION.md`.

