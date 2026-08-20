"""Model training, threshold selection, and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import MODEL_FEATURES


@dataclass(frozen=True)
class TemporalPartitions:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_split(windows: pd.DataFrame, train_fraction: float, validation_fraction: float) -> TemporalPartitions:
    """Split complete timestamps chronologically to avoid adjacent-window leakage."""
    times = np.array(sorted(pd.to_datetime(windows["window_start"]).unique()))
    train_end = max(1, int(len(times) * train_fraction))
    validation_end = max(train_end + 1, int(len(times) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(times) - 1)
    train_times = set(times[:train_end])
    validation_times = set(times[train_end:validation_end])
    test_times = set(times[validation_end:])
    return TemporalPartitions(
        windows[windows["window_start"].isin(train_times)].copy(),
        windows[windows["window_start"].isin(validation_times)].copy(),
        windows[windows["window_start"].isin(test_times)].copy(),
    )


def build_random_forest(seed: int) -> Pipeline:
    transformer = ColumnTransformer(
        (("service", OneHotEncoder(handle_unknown="ignore"), ["service"]),),
        remainder="passthrough",
    )
    return Pipeline(
        (
            ("preprocess", transformer),
            ("model", RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)),
        )
    )


def build_isolation_forest(seed: int, contamination: float) -> Pipeline:
    numeric = [column for column in MODEL_FEATURES if column != "service"]
    transformer = ColumnTransformer(
        (
            ("service", OneHotEncoder(handle_unknown="ignore"), ["service"]),
            ("numeric", StandardScaler(), numeric),
        )
    )
    return Pipeline(
        (
            ("preprocess", transformer),
            ("model", IsolationForest(n_estimators=300, contamination=contamination, random_state=seed, n_jobs=-1)),
        )
    )


def select_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    """Choose the validation threshold that maximises anomaly F1."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 199)))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, predictions, average="binary", zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), float(f1)
    return best_threshold


def evaluate_predictions(
    model_name: str,
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> tuple[dict, pd.DataFrame]:
    actual = frame["anomaly_label"].astype(int).to_numpy()
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=(0, 1)).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average="binary", zero_division=0)
    metrics = {
        "model": model_name,
        "threshold": float(threshold),
        "test_windows": int(len(frame)),
        "anomalous_windows": int(actual.sum()),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "pr_auc": float(average_precision_score(actual, scores)) if len(np.unique(actual)) > 1 else 0.0,
    }
    predictions = frame[["window_start", "window_end", "service", "incident_id", "anomaly_type", "anomaly_label"]].copy()
    predictions["model"] = model_name
    predictions["anomaly_score"] = scores
    predictions["prediction"] = predicted
    return metrics, predictions


def incident_metrics(predictions: pd.DataFrame) -> dict:
    incidents = predictions[predictions["incident_id"].notna()].copy()
    if incidents.empty:
        return {"incident_count": 0, "detected_incidents": 0, "incident_recall": 0.0, "mean_detection_delay_seconds": None}
    summaries = []
    for incident_id, rows in incidents.groupby("incident_id"):
        positives = rows[rows["prediction"] == 1]
        detected = not positives.empty
        first_anomalous_window = pd.to_datetime(rows["window_start"]).min()
        # A window prediction is only available when that window closes.
        first_detection = pd.to_datetime(positives["window_end"]).min() if detected else pd.NaT
        delay = (first_detection - first_anomalous_window).total_seconds() if detected else None
        summaries.append((incident_id, detected, delay))
    delays = [item[2] for item in summaries if item[2] is not None]
    return {
        "incident_count": len(summaries),
        "detected_incidents": sum(item[1] for item in summaries),
        "incident_recall": sum(item[1] for item in summaries) / len(summaries),
        "mean_detection_delay_seconds": float(np.mean(delays)) if delays else None,
    }
