"""
Training utilities for the real-time accelerometer accident detector.

The pipeline:
1. Windows raw accelerometer streams via `AccelerometerFeatureEngineer`.
2. Extracts statistical, derived, and spectral features.
3. Fits an XGBoost classifier tuned for high recall without sacrificing precision.
4. Provides evaluation summaries, threshold optimisation, and export helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.feature_engineering.accelerometer_feature_engineer import (
    AccelerometerFeatureEngineer,
)


@dataclass
class TrainingArtifacts:
    """Container storing fitted pipeline along with metadata."""

    pipeline: Pipeline
    feature_columns: list
    metrics: Dict[str, float]
    threshold: float


class AccelerometerAccidentDetector:
    """
    End-to-end trainer for binary accident detection.

    Parameters
    ----------
    sampling_rate : int
        Sampling frequency in Hz.
    window_size : int
        Samples per segment.
    step_size : Optional[int]
        Hop length between windows.
    model_params : Optional[Dict]
        Overrides for XGBoost hyperparameters.
    target_recall : float
        Minimum recall to prioritise when selecting thresholds.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        sampling_rate: int = 50,
        window_size: int = 100,
        step_size: Optional[int] = None,
        label_col: str = "accident",
        model_params: Optional[Dict] = None,
        target_recall: float = 0.9,
        random_state: int = 42,
    ) -> None:
        self.feature_engineer = AccelerometerFeatureEngineer(
            sampling_rate=sampling_rate,
            window_size=window_size,
            step_size=step_size,
            label_col=label_col,
        )
        self.random_state = random_state
        self.target_recall = target_recall
        self.label_col = label_col
        self.model_params = model_params or {
            "n_estimators": 600,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "min_child_weight": 4,
            "gamma": 0.5,
            "reg_lambda": 1.5,
            "reg_alpha": 0.1,
            "eval_metric": "logloss",
            "tree_method": "hist",
        }
        self.artifacts: Optional[TrainingArtifacts] = None

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(
        self,
        df: pd.DataFrame,
        sensor_cols=None,
        timestamp_col: str = "timestamp",
        vehicle_id_col: str = "vehicle_id",
        severity_col: Optional[str] = "event_severity",
        metadata_cols=None,
        test_size: float = 0.2,
    ) -> TrainingArtifacts:
        """Train the detector and persist artifacts in memory."""

        features_df = self.feature_engineer.transform(
            df=df,
            sensor_cols=sensor_cols or ("accel_x", "accel_y", "accel_z"),
            timestamp_col=timestamp_col,
            vehicle_id_col=vehicle_id_col,
            severity_col=severity_col,
            metadata_cols=metadata_cols,
        )
        self.last_features_ = features_df.copy()

        non_feature_cols = {
            "label",
            "vehicle_id",
            "window_index",
            "window_start_ts",
            "window_end_ts",
        }
        feature_cols = [col for col in features_df.columns if col not in non_feature_cols]

        X = features_df[feature_cols].values
        y = features_df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state,
        )

        model_params = self._apply_class_weight(y_train)
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", XGBClassifier(**model_params)),
            ]
        )
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = self._collect_metrics(y_test, y_pred, y_proba)
        threshold_info = self._optimise_threshold(y_test, y_proba)
        metrics.update(threshold_info)

        self.artifacts = TrainingArtifacts(
            pipeline=pipe,
            feature_columns=feature_cols,
            metrics=metrics,
            threshold=threshold_info["optimal_threshold"],
        )
        return self.artifacts

    # ------------------------------------------------------------------ #
    # Evaluation + CV
    # ------------------------------------------------------------------ #
    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, n_splits: int = 5
    ) -> Dict[str, float]:
        """Perform stratified cross-validation on already engineered features."""

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        auc_scores: list = []
        recall_scores: list = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            params = self._apply_class_weight(y_train)
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            auc_scores.append(roc_auc_score(y_val, y_proba))
            true_positives = np.sum((y_pred == 1) & (y_val == 1))
            recall_scores.append(true_positives / max(1, np.sum(y_val == 1)))

        return {
            "cv_auc_mean": float(np.mean(auc_scores)),
            "cv_auc_std": float(np.std(auc_scores)),
            "cv_recall_mean": float(np.mean(recall_scores)),
            "cv_recall_std": float(np.std(recall_scores)),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_model(self, path: str | Path) -> None:
        """Persist trained pipeline using joblib."""
        if not self.artifacts:
            raise RuntimeError("Call fit() before saving the model.")
        joblib.dump(self.artifacts.pipeline, path)

    def save_metrics(self, path: str | Path) -> None:
        """Write evaluation metrics to JSON."""
        if not self.artifacts:
            raise RuntimeError("Call fit() before saving metrics.")
        Path(path).write_text(json.dumps(self.artifacts.metrics, indent=2))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _apply_class_weight(self, y: np.ndarray) -> Dict:
        params = dict(self.model_params)
        pos = y.sum()
        neg = len(y) - pos
        if pos == 0:
            params["scale_pos_weight"] = 1.0
        else:
            params["scale_pos_weight"] = max(1.0, float(neg / pos))
        params["random_state"] = self.random_state
        return params

    def _collect_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, float]:
        report = classification_report(
            y_true, y_pred, target_names=["normal", "accident"], output_dict=True
        )
        cm = confusion_matrix(y_true, y_pred).tolist()
        return {
            "accuracy": float(report["accuracy"]),
            "precision": float(report["accident"]["precision"]),
            "recall": float(report["accident"]["recall"]),
            "f1": float(report["accident"]["f1-score"]),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "confusion_matrix": cm,
        }

    def _optimise_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
        threshold_candidates = np.concatenate(([0.0], thresholds))
        best_idx = int(np.argmax(f1_scores))
        optimal_threshold = float(threshold_candidates[min(best_idx, len(threshold_candidates) - 1)])

        target_idx = np.where(recall >= self.target_recall)[0]
        if target_idx.size:
            idx = target_idx[0]
            recall_threshold = float(threshold_candidates[min(idx, len(threshold_candidates) - 1)])
            recall_precision = float(precision[idx])
            recall_recall = float(recall[idx])
        else:
            recall_threshold = optimal_threshold
            recall_precision = float(precision[best_idx])
            recall_recall = float(recall[best_idx])

        return {
            "optimal_threshold": optimal_threshold,
            "optimal_precision": float(precision[best_idx]),
            "optimal_recall": float(recall[best_idx]),
            "recall_threshold": recall_threshold,
            "recall_precision": recall_precision,
            "recall_recall": recall_recall,
        }


