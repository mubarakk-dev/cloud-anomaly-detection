"""Run the complete reproducible offline academic experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd

from .config import ExperimentConfig
from .features import MODEL_FEATURES, aggregate_windows
from .modelling import (
    build_isolation_forest,
    build_random_forest,
    evaluate_predictions,
    incident_metrics,
    select_threshold,
    temporal_split,
)
from .telemetry import generate_telemetry


def run(config: ExperimentConfig, output_root: Path) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    for directory in ("datasets", "metrics", "predictions"):
        (output_root / directory).mkdir(exist_ok=True)
    model_root = output_root.parent / "models"
    model_root.mkdir(parents=True, exist_ok=True)

    events, incidents = generate_telemetry(config)
    windows = aggregate_windows(events, config.window_seconds)
    partitions = temporal_split(windows, config.train_fraction, config.validation_fraction)

    events.to_csv(output_root / "datasets" / "synthetic_telemetry.csv", index=False)
    incidents.to_csv(output_root / "datasets" / "incident_manifest.csv", index=False)
    windows.to_csv(output_root / "datasets" / "telemetry_windows.csv", index=False)

    feature_columns = list(MODEL_FEATURES)
    random_forest = build_random_forest(config.seed)
    random_forest.fit(partitions.train[feature_columns], partitions.train["anomaly_label"])
    rf_validation_scores = random_forest.predict_proba(partitions.validation[feature_columns])[:, 1]
    rf_threshold = select_threshold(partitions.validation["anomaly_label"], rf_validation_scores)
    rf_test_scores = random_forest.predict_proba(partitions.test[feature_columns])[:, 1]
    rf_metrics, rf_predictions = evaluate_predictions("random_forest", partitions.test, rf_test_scores, rf_threshold)
    rf_metrics.update(incident_metrics(rf_predictions))

    normal_training = partitions.train[partitions.train["anomaly_label"] == 0]
    isolation_forest = build_isolation_forest(config.seed, config.isolation_contamination)
    isolation_forest.fit(normal_training[feature_columns])
    iso_validation_scores = -isolation_forest.decision_function(partitions.validation[feature_columns])
    iso_threshold = select_threshold(partitions.validation["anomaly_label"], iso_validation_scores)
    iso_test_scores = -isolation_forest.decision_function(partitions.test[feature_columns])
    iso_metrics, iso_predictions = evaluate_predictions("isolation_forest", partitions.test, iso_test_scores, iso_threshold)
    iso_metrics.update(incident_metrics(iso_predictions))

    joblib.dump(
        {"pipeline": random_forest, "threshold": rf_threshold, "features": feature_columns, "window_seconds": config.window_seconds},
        model_root / "window_random_forest.joblib",
    )
    joblib.dump(
        {"pipeline": isolation_forest, "threshold": iso_threshold, "features": feature_columns, "window_seconds": config.window_seconds},
        model_root / "window_isolation_forest.joblib",
    )

    predictions = pd.concat((rf_predictions, iso_predictions), ignore_index=True)
    predictions.to_csv(output_root / "predictions" / "test_predictions.csv", index=False)
    metrics = pd.DataFrame((rf_metrics, iso_metrics))
    metrics.to_csv(output_root / "metrics" / "model_comparison.csv", index=False)
    (output_root / "metrics" / "experiment_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--window-seconds", type=int, default=30)
    parser.add_argument("--incident-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    config = ExperimentConfig(
        duration_seconds=args.duration_seconds,
        window_seconds=args.window_seconds,
        incident_count=args.incident_count,
        seed=args.seed,
    )
    metrics = run(config, args.output_root)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
