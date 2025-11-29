"""
Feature generation utilities for high-frequency accelerometer windows.

This module converts raw tri-axial accelerometer readings into fixed-length
window-level feature vectors that can be consumed by downstream classifiers.
It supports the statistics outlined in the accident-detection requirements,
including advanced descriptors such as jerk, energy, cross-axis correlation
and FFT-derived spectral summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


@dataclass
class WindowMetadata:
    """Container for metadata describing a single time window."""

    vehicle_id: str
    window_index: int
    start_timestamp: float
    end_timestamp: float
    event_ids: List
    severity: Optional[float] = None


class AccelerometerFeatureEngineer:
    """
    Transforms streaming accelerometer readings into engineered window features.

    Parameters
    ----------
    sampling_rate : int
        Sampling frequency in Hz (samples per second). Default: 50 Hz.
    window_size : int
        Number of samples per fixed window. Default: 100.
    step_size : Optional[int]
        Hop length between consecutive windows. Defaults to 50% overlap.
    label_col : str
        Column name representing binary event labels (0 = normal, 1 = accident).
    """

    def __init__(
        self,
        sampling_rate: int = 50,
        window_size: int = 100,
        step_size: Optional[int] = None,
        label_col: str = "accident",
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.step_size = step_size or window_size // 2
        self.label_col = label_col

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def transform(
        self,
        df: pd.DataFrame,
        sensor_cols: Sequence[str] = ("accel_x", "accel_y", "accel_z"),
        timestamp_col: str = "timestamp",
        vehicle_id_col: str = "vehicle_id",
        severity_col: Optional[str] = "event_severity",
        metadata_cols: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        Slice raw samples into windows and compute engineered features.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing raw sensor readings.
        sensor_cols : Sequence[str]
            Names of accelerometer axes.
        timestamp_col : str
            Column containing ordered timestamps (datetime or numeric).
        vehicle_id_col : str
            Identifier used to segment streams per vehicle/device.
        severity_col : Optional[str]
            Optional column describing severity metadata.
        metadata_cols : Optional[Iterable[str]]
            Additional metadata columns to aggregate (e.g., event_id).

        Returns
        -------
        pd.DataFrame
            One row per window with engineered features and labels.
        """

        missing_cols = [c for c in sensor_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing sensor columns: {missing_cols}")
        if self.label_col not in df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in dataframe")

        working_df = df.copy()
        if timestamp_col not in working_df.columns:
            # Create a synthetic timestamp in seconds relative to start.
            working_df[timestamp_col] = np.arange(len(working_df)) / float(
                self.sampling_rate
            )

        if vehicle_id_col not in working_df.columns:
            working_df[vehicle_id_col] = "vehicle_0"

        working_df = working_df.sort_values([vehicle_id_col, timestamp_col])
        metadata_cols = list(metadata_cols or [])
        metadata_cols = [col for col in metadata_cols if col in working_df.columns]

        windows: List[Dict[str, float]] = []
        for vehicle_id, group in working_df.groupby(vehicle_id_col):
            group = group.reset_index(drop=True)
            num_samples = len(group)
            if num_samples < self.window_size:
                continue

            for start_idx in range(0, num_samples - self.window_size + 1, self.step_size):
                end_idx = start_idx + self.window_size
                window = group.iloc[start_idx:end_idx]
                feature_row = self._compute_window_features(
                    window=window,
                    sensor_cols=sensor_cols,
                    metadata_cols=metadata_cols,
                    severity_col=severity_col,
                )

                feature_row["vehicle_id"] = vehicle_id
                feature_row["window_index"] = start_idx // self.step_size
                feature_row["window_start_ts"] = window[timestamp_col].iloc[0]
                feature_row["window_end_ts"] = window[timestamp_col].iloc[-1]
                feature_row["label"] = int(window[self.label_col].any())
                feature_row["positive_ratio"] = window[self.label_col].mean()

                windows.append(feature_row)

        if not windows:
            raise ValueError("No windows were generated — check window_size/step_size.")

        features_df = pd.DataFrame(windows)
        feature_cols = [
            col
            for col in features_df.columns
            if col
            not in {
                "label",
                "vehicle_id",
                "window_index",
                "window_start_ts",
                "window_end_ts",
                "positive_ratio",
            }
        ]
        features_df[feature_cols] = features_df[feature_cols].replace(
            [np.inf, -np.inf], np.nan
        )
        features_df[feature_cols] = features_df[feature_cols].fillna(0.0)
        return features_df

    # --------------------------------------------------------------------- #
    # Feature helpers
    # --------------------------------------------------------------------- #
    def _compute_window_features(
        self,
        window: pd.DataFrame,
        sensor_cols: Sequence[str],
        metadata_cols: Sequence[str],
        severity_col: Optional[str],
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}

        axis_stats = {
            axis: self._axis_statistics(window[axis].values, prefix=axis)
            for axis in sensor_cols
        }
        for axis_dict in axis_stats.values():
            features.update(axis_dict)

        magnitude = self._vector_magnitude(window[list(sensor_cols)].values)
        features.update(self._series_statistics(magnitude, prefix="magnitude"))

        jerk_axes = {
            axis: self._jerk_statistics(window[axis].values, prefix=f"{axis}_jerk")
            for axis in sensor_cols
        }
        jerk_axes["magnitude"] = self._jerk_statistics(magnitude, prefix="magnitude_jerk")
        for jerk_dict in jerk_axes.values():
            features.update(jerk_dict)

        features.update(self._correlation_features(window, sensor_cols))

        fft_features = {
            axis: self._fft_features(window[axis].values, prefix=f"{axis}_fft")
            for axis in sensor_cols
        }
        fft_features["magnitude_fft"] = self._fft_features(
            magnitude, prefix="magnitude_fft"
        )
        for fft_dict in fft_features.values():
            features.update(fft_dict)

        if metadata_cols:
            for meta_col in metadata_cols:
                meta_values = window[meta_col]
                features[f"{meta_col}_last"] = meta_values.iloc[-1]
                mode_values = meta_values.mode(dropna=True)
                if not mode_values.empty:
                    features[f"{meta_col}_mode"] = mode_values.iloc[0]
                else:
                    features[f"{meta_col}_mode"] = meta_values.iloc[-1]

        if severity_col and severity_col in window.columns:
            severity_series = window[severity_col]
            if severity_series.dtype == "O":
                severity_mapping = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
                severity_numeric = severity_series.map(severity_mapping).fillna(0)
            else:
                severity_numeric = severity_series.fillna(0)

            features["severity_max"] = severity_numeric.max()
            features["severity_mean"] = severity_numeric.mean()

        features["samples_in_window"] = len(window)
        return features

    def _axis_statistics(self, series: np.ndarray, prefix: str) -> Dict[str, float]:
        return self._series_statistics(series, prefix=prefix)

    def _series_statistics(self, series: np.ndarray, prefix: str) -> Dict[str, float]:
        if len(series) == 0:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_std": 0.0,
                f"{prefix}_max": 0.0,
                f"{prefix}_min": 0.0,
                f"{prefix}_range": 0.0,
                f"{prefix}_ptp": 0.0,
                f"{prefix}_energy": 0.0,
                f"{prefix}_skew": 0.0,
                f"{prefix}_kurtosis": 0.0,
                f"{prefix}_zcr": 0.0,
            }

        stats = {
            f"{prefix}_mean": float(np.mean(series)),
            f"{prefix}_std": float(np.std(series)),
            f"{prefix}_max": float(np.max(series)),
            f"{prefix}_min": float(np.min(series)),
        }
        stats[f"{prefix}_range"] = stats[f"{prefix}_max"] - stats[f"{prefix}_min"]
        stats[f"{prefix}_ptp"] = float(np.ptp(series))
        stats[f"{prefix}_energy"] = float(np.sum(series**2))
        stats[f"{prefix}_skew"] = float(skew(series))
        stats[f"{prefix}_kurtosis"] = float(kurtosis(series))
        stats[f"{prefix}_zcr"] = float(self._zero_crossing_rate(series))
        return stats

    def _jerk_statistics(self, series: np.ndarray, prefix: str) -> Dict[str, float]:
        if len(series) < 2:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_std": 0.0,
                f"{prefix}_max": 0.0,
            }
        jerk = np.diff(series) * self.sampling_rate
        return {
            f"{prefix}_mean": float(np.mean(jerk)),
            f"{prefix}_std": float(np.std(jerk)),
            f"{prefix}_max": float(np.max(np.abs(jerk))),
        }

    def _fft_features(self, series: np.ndarray, prefix: str) -> Dict[str, float]:
        if len(series) < 2:
            return {
                f"{prefix}_dominant_freq": 0.0,
                f"{prefix}_dominant_power": 0.0,
                f"{prefix}_spectral_energy": 0.0,
                f"{prefix}_spectral_entropy": 0.0,
            }

        spectrum = np.abs(np.fft.rfft(series))
        freqs = np.fft.rfftfreq(len(series), d=1.0 / self.sampling_rate)
        spectrum[0] = 0  # ignore DC offset for dominance/entropy

        dominant_idx = int(np.argmax(spectrum)) if np.any(spectrum) else 0
        spectral_energy = float(np.sum(spectrum**2))
        power_distribution = spectrum / (np.sum(spectrum) + 1e-12)
        spectral_entropy = float(
            -np.sum(power_distribution * np.log(power_distribution + 1e-12))
        )

        return {
            f"{prefix}_dominant_freq": float(freqs[dominant_idx]),
            f"{prefix}_dominant_power": float(spectrum[dominant_idx]),
            f"{prefix}_spectral_energy": spectral_energy,
            f"{prefix}_spectral_entropy": spectral_entropy,
        }

    def _correlation_features(
        self, window: pd.DataFrame, sensor_cols: Sequence[str]
    ) -> Dict[str, float]:
        correlations: Dict[str, float] = {}
        for i, col_i in enumerate(sensor_cols):
            for col_j in sensor_cols[i + 1 :]:
                corr_value = window[col_i].corr(window[col_j])
                corr_key = f"corr_{col_i}_{col_j}"
                correlations[corr_key] = float(corr_value) if not np.isnan(corr_value) else 0.0
        return correlations

    @staticmethod
    def _zero_crossing_rate(series: np.ndarray) -> float:
        if len(series) < 2:
            return 0.0
        centered = series - np.mean(series)
        crossings = np.where(np.diff(np.signbit(centered)))[0]
        return float(len(crossings)) / (len(series) - 1)

    @staticmethod
    def _vector_magnitude(values: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum(values**2, axis=1))


