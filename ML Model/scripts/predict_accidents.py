"""
Inference script for the trained accelerometer accident detector.

Usage:
    python scripts/predict_accidents.py \
        --model-path models/accelerometer_accident_detector.pkl \
        --data-path new_sensor_data.csv \
        --output-path predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Ensure src/ modules are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.feature_engineering import AccelerometerFeatureEngineer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on accelerometer data using trained model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "accelerometer_accident_detector.pkl"),
        help="Path to trained model (.pkl file).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="CSV file with accelerometer readings to predict.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(PROJECT_ROOT / "output" / "predictions.csv"),
        help="Output CSV with predictions.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=50,
        help="Sensor sampling rate in Hz (must match training).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Samples per window (must match training).",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=None,
        help="Hop length between windows (must match training).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold (0-1). If None, uses model's optimal threshold.",
    )
    parser.add_argument(
        "--sensor-cols",
        type=str,
        nargs=3,
        default=["accel_x", "accel_y", "accel_z"],
        help="Names of accelerometer columns.",
    )
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default="timestamp",
        help="Timestamp column name.",
    )
    parser.add_argument(
        "--vehicle-id-col",
        type=str,
        default="vehicle_id",
        help="Vehicle identifier column.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="accident",
        help="Label column (can be dummy if unlabeled data).",
    )
    return parser.parse_args()


def load_model(path: str) -> Pipeline:
    """Load trained pipeline from disk."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def load_metrics(path: str) -> dict:
    """Load metrics JSON to get optimal threshold."""
    import json

    metrics_path = Path(path)
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {}


def main() -> None:
    args = parse_args()

    print(f"[+] Loading model from {args.model_path}")
    pipeline = load_model(args.model_path)

    # Try to load metrics for optimal threshold
    metrics_path = Path(args.model_path).parent / "accelerometer_metrics.json"
    if metrics_path.exists():
        metrics = load_metrics(str(metrics_path))
        default_threshold = metrics.get("optimal_threshold", 0.5)
        print(f"[+] Loaded optimal threshold: {default_threshold:.3f}")
    else:
        default_threshold = 0.5
        print(f"[!] Metrics file not found, using default threshold: {default_threshold}")

    threshold = args.threshold if args.threshold is not None else default_threshold

    print(f"[+] Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)

    # Handle label column (create dummy if missing)
    if args.label_col not in df.columns:
        print(f"[!] Label column '{args.label_col}' not found, creating dummy column")
        df[args.label_col] = 0

    # Handle timestamp
    if args.timestamp_col not in df.columns:
        print(f"[!] Timestamp column not found, generating synthetic timestamps")
        df[args.timestamp_col] = np.arange(len(df)) / float(args.sampling_rate)

    # Handle vehicle_id
    if args.vehicle_id_col not in df.columns:
        print(f"[!] Vehicle ID column not found, using default")
        df[args.vehicle_id_col] = "vehicle_0"

    print(f"[+] Loaded {len(df):,} samples")

    # Engineer features
    print("[+] Engineering features...")
    engineer = AccelerometerFeatureEngineer(
        sampling_rate=args.sampling_rate,
        window_size=args.window_size,
        step_size=args.step_size or args.window_size // 2,
        label_col=args.label_col,
    )

    features_df = engineer.transform(
        df=df,
        sensor_cols=tuple(args.sensor_cols),
        timestamp_col=args.timestamp_col,
        vehicle_id_col=args.vehicle_id_col,
    )

    print(f"[+] Generated {len(features_df):,} windows")

    # Extract feature columns (same logic as training)
    non_feature_cols = {
        "label",
        "vehicle_id",
        "window_index",
        "window_start_ts",
        "window_end_ts",
        "positive_ratio",
    }
    feature_cols = [col for col in features_df.columns if col not in non_feature_cols]

    X = features_df[feature_cols].values

    # Predict
    print("[+] Running predictions...")
    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    # Add predictions to dataframe
    features_df["accident_probability"] = probabilities
    features_df["predicted_accident"] = predictions

    # Summary statistics
    n_accidents = predictions.sum()
    n_windows = len(predictions)
    print(f"[+] Predictions complete:")
    print(f"    Total windows: {n_windows:,}")
    print(f"    Predicted accidents: {n_accidents:,} ({100*n_accidents/n_windows:.1f}%)")
    print(f"    Average probability: {probabilities.mean():.3f}")
    print(f"    Max probability: {probabilities.max():.3f}")

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"[+] Saved predictions to {output_path}")

    # Show high-probability windows
    high_prob = features_df[features_df["accident_probability"] >= 0.7]
    if len(high_prob) > 0:
        print(f"\n[!] {len(high_prob)} windows with probability >= 0.7:")
        print(high_prob[["vehicle_id", "window_start_ts", "accident_probability", "predicted_accident"]].head(10))


if __name__ == "__main__":
    main()

