# Accelerometer Accident Detection Pipeline

This guide describes the real-time accident detection model that ingests raw tri-axial accelerometer streams, engineers window-level features, and trains an XGBoost classifier tuned for high precision and recall.

## 1. Data Requirements

- **Sampling rate**: default 50 Hz (configurable via CLI).
- **Windowing**: fixed 100-sample windows (2 seconds at 50 Hz) with 50% overlap.
- **Sensors**: `accel_x`, `accel_y`, `accel_z` (m/s²).
- **Labels**: binary `accident` flag (0 = normal, 1 = crash). Window label is positive if any sample inside the window is positive.
- **Metadata**:
  - `timestamp` (datetime or monotonic numeric) – used to preserve ordering.
  - `vehicle_id` – isolates per-vehicle streams.
  - `event_id` and `event_severity` (optional) – aggregated for downstream analytics.
- **Coverage**: collect diverse driving manoeuvres (steady cruising, harsh braking, potholes, collisions) to balance classes. The trainer auto-adjusts class weights based on observed imbalance.

## 2. Feature Engineering

Implemented in `src/feature_engineering/accelerometer_feature_engineer.py`.

| Category | Features |
| --- | --- |
| **Per-axis stats** | mean, std, min, max, range, peak-to-peak, energy, skewness, kurtosis, zero-crossing rate |
| **Derived magnitude** | vector magnitude per sample, plus same statistics as above |
| **Jerk (∆acc/∆t)** | mean, std, max absolute jerk per axis and on magnitude |
| **Energy** | Sum of squared acceleration for each axis and magnitude |
| **Correlation** | Pearson correlations (xy, xz, yz) |
| **Spectral (FFT)** | Dominant frequency, dominant power, spectral energy, spectral entropy per axis and magnitude |
| **Metadata aggregates** | Last/mode values for any supplied metadata columns, severity mean/max, window timing, positive sample ratio |

All features default to zero when undefined (e.g., constant signals). Infinite values are sanitised.

## 3. Model Development

Implemented in `src/model_training/accelerometer_accident_detector.py`.

- **Model**: `XGBClassifier` (tree_method=`hist`) wrapped inside a `StandardScaler` pipeline.
- **Key hyperparameters**:
  - `n_estimators=600`, `learning_rate=0.05`
  - `max_depth=6`, `min_child_weight=4`
  - `subsample=0.85`, `colsample_bytree=0.8`
  - `gamma=0.5`, `reg_lambda=1.5`, `reg_alpha=0.1`
  - `scale_pos_weight` auto-computed as `negatives / positives`
  - `eval_metric="logloss"`
- **Evaluation**:
  - Hold-out split (default 80/20) with stratification.
  - Metrics: accuracy, precision, recall, F1 (positive class), ROC-AUC, confusion matrix.
  - Precision–recall sweep selects (a) best F1 threshold, (b) earliest threshold achieving configurable recall target (default ≥0.90).
  - Optional stratified k-fold cross-validation on engineered feature matrices.

## 4. Training Script

Use `scripts/train_accelerometer_accident_detector.py`:

```bash
python scripts/train_accelerometer_accident_detector.py \
  --data-path scripts/synthetic_accident_sensor_data.csv \
  --sampling-rate 50 \
  --window-size 100 \
  --label-col accident \
  --timestamp-col timestamp \
  --vehicle-id-col vehicle_id
```

Outputs:

- Model artifact → `models/accelerometer_accident_detector.pkl`
- Metrics JSON → `output/accelerometer_metrics.json`
- Optional engineered features parquet → `output/accelerometer_features.parquet`

Override CLI flags to point at production telemetry, tweak window sizes, or adjust metadata column names.

## 5. Real-Time Inference Considerations

- Stream accelerometer readings into a ring buffer sized to `window_size`.
- Compute features incrementally (rolling stats, FFT via overlap-save) for low latency.
- Apply the trained pipeline and compare accident probability to the stored optimal threshold (or the recall-first threshold when safety-critical).
- Expose vehicle ID and timestamps from the metadata fields to trigger downstream alerting and logging workflows.

## 6. Next Steps

1. **Calibration**: capture labelled data from field devices to fine-tune thresholds per fleet.
2. **Edge deployment**: convert the trained booster to ONNX or Treelite for embedded targets.
3. **Model monitoring**: log prediction probabilities, latency, and class distribution to detect drift.
4. **Explainability**: enable SHAP feature attributions to help safety engineers audit triggers.
5. **Sensor fusion**: extend the feature engineer to incorporate gyroscope and speed channels when available.


