"""Individual-event baselines for comparison with behavioural windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ExperimentConfig
from .modelling import evaluate_predictions, incident_metrics, select_threshold
from .telemetry import generate_telemetry


EVENT_FEATURES = (
    "service", "event_type", "response_time_ms", "cpu_usage", "memory_usage", "status_code", "log_level",
)


def _partitions(events: pd.DataFrame, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = events.copy()
    frame["window_start"] = pd.to_datetime(frame["timestamp"])
    frame["window_end"] = frame["window_start"]
    frame["anomaly_label"] = frame["label"]
    unique_times = np.array(sorted(frame["window_start"].unique()))
    train_end = int(len(unique_times) * config.train_fraction)
    validation_end = int(len(unique_times) * (config.train_fraction + config.validation_fraction))
    return (
        frame[frame["window_start"].isin(set(unique_times[:train_end]))].copy(),
        frame[frame["window_start"].isin(set(unique_times[train_end:validation_end]))].copy(),
        frame[frame["window_start"].isin(set(unique_times[validation_end:]))].copy(),
    )


def _random_forest(seed: int) -> Pipeline:
    categorical = ["service", "event_type", "status_code", "log_level"]
    transformer = ColumnTransformer(
        (("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),),
        remainder="passthrough",
    )
    return Pipeline((
        ("preprocess", transformer),
        ("model", RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)),
    ))


def _isolation_forest(seed: int, contamination: float) -> Pipeline:
    categorical = ["service", "event_type", "status_code", "log_level"]
    numeric = ["response_time_ms", "cpu_usage", "memory_usage"]
    transformer = ColumnTransformer((
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", StandardScaler(), numeric),
    ))
    return Pipeline((
        ("preprocess", transformer),
        ("model", IsolationForest(n_estimators=300, contamination=contamination, random_state=seed, n_jobs=-1)),
    ))


def run_event_baseline(config: ExperimentConfig, output_root: Path) -> pd.DataFrame:
    events, _ = generate_telemetry(config)
    train, validation, test = _partitions(events, config)

    rf = _random_forest(config.seed)
    rf.fit(train[list(EVENT_FEATURES)], train["anomaly_label"])
    rf_validation = rf.predict_proba(validation[list(EVENT_FEATURES)])[:, 1]
    rf_threshold = select_threshold(validation["anomaly_label"], rf_validation)
    rf_scores = rf.predict_proba(test[list(EVENT_FEATURES)])[:, 1]
    rf_metrics, rf_predictions = evaluate_predictions("random_forest", test, rf_scores, rf_threshold)
    rf_metrics.update(incident_metrics(rf_predictions))

    normal_train = train[train["anomaly_label"] == 0]
    iso = _isolation_forest(config.seed, config.isolation_contamination)
    iso.fit(normal_train[list(EVENT_FEATURES)])
    iso_validation = -iso.decision_function(validation[list(EVENT_FEATURES)])
    iso_threshold = select_threshold(validation["anomaly_label"], iso_validation)
    iso_scores = -iso.decision_function(test[list(EVENT_FEATURES)])
    iso_metrics, iso_predictions = evaluate_predictions("isolation_forest", test, iso_scores, iso_threshold)
    iso_metrics.update(incident_metrics(iso_predictions))

    output_root.mkdir(parents=True, exist_ok=True)
    pd.concat((rf_predictions, iso_predictions), ignore_index=True).to_csv(output_root / "event_predictions.csv", index=False)
    metrics = pd.DataFrame((rf_metrics, iso_metrics))
    metrics.to_csv(output_root / "event_metrics.csv", index=False)
    return metrics
