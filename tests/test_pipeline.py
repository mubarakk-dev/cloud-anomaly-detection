from dataclasses import replace

import joblib
import numpy as np
import pandas as pd

from src.config import ExperimentConfig, SERVICES
from src.features import aggregate_windows
from src.features import MODEL_FEATURES
from src.modelling import build_random_forest, temporal_split
from src.stream_detector import StreamingWindowDetector
from src.telemetry import generate_incidents, generate_telemetry


def small_config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=7,
        duration_seconds=300,
        window_seconds=30,
        incident_count=3,
        incident_min_seconds=20,
        incident_max_seconds=40,
    )


def test_generation_is_reproducible():
    config = small_config()
    first_events, first_incidents = generate_telemetry(config)
    second_events, second_incidents = generate_telemetry(config)
    pd.testing.assert_frame_equal(first_events, second_events)
    pd.testing.assert_frame_equal(first_incidents, second_incidents)


def test_one_event_exists_for_every_service_and_tick():
    config = small_config()
    events, _ = generate_telemetry(config)
    expected_ticks = config.duration_seconds // config.tick_seconds
    assert len(events) == expected_ticks * len(SERVICES)
    assert events.groupby("timestamp")["service"].nunique().eq(len(SERVICES)).all()


def test_incidents_only_affect_their_declared_service():
    config = small_config()
    incidents = generate_incidents(config)
    events, manifest = generate_telemetry(config, incidents=incidents)
    anomalous = events[events["label"] == 1]
    service_by_incident = manifest.set_index("incident_id")["service"]
    assert anomalous.apply(lambda row: row["service"] == service_by_incident[row["incident_id"]], axis=1).all()


def test_incidents_do_not_cross_temporal_split_boundaries():
    config = small_config()
    incidents = generate_incidents(config)
    boundaries = (
        config.duration_seconds * config.train_fraction,
        config.duration_seconds * (config.train_fraction + config.validation_fraction),
    )
    assert all(
        not (incident.start_offset_seconds < boundary < incident.end_offset_seconds)
        for incident in incidents
        for boundary in boundaries
    )


def test_each_partition_contains_every_anomaly_type_when_capacity_allows():
    from src.config import ANOMALY_TYPES

    config = ExperimentConfig(
        seed=8,
        duration_seconds=1200,
        window_seconds=30,
        incident_count=15,
        incident_min_seconds=20,
        incident_max_seconds=40,
    )
    incidents = generate_incidents(config)
    boundaries = (config.duration_seconds * config.train_fraction, config.duration_seconds * (config.train_fraction + config.validation_fraction))
    groups = (
        [item for item in incidents if item.start_offset_seconds < boundaries[0]],
        [item for item in incidents if boundaries[0] < item.start_offset_seconds < boundaries[1]],
        [item for item in incidents if item.start_offset_seconds > boundaries[1]],
    )
    for group in groups:
        if len(group) >= len(ANOMALY_TYPES):
            assert set(ANOMALY_TYPES).issubset({item.anomaly_type for item in group})


def test_window_features_and_counts_are_consistent():
    config = replace(small_config(), incident_count=1)
    events, _ = generate_telemetry(config)
    windows = aggregate_windows(events, config.window_seconds)
    assert windows["log_count"].eq(config.window_seconds // config.tick_seconds).all()
    assert windows["error_rate"].between(0, 1).all()
    assert windows["warn_rate"].between(0, 1).all()
    assert windows["error_log_rate"].between(0, 1).all()


def test_model_features_do_not_include_ground_truth():
    from src.features import MODEL_FEATURES

    forbidden = {"label", "anomaly_label", "incident_id", "anomaly_type"}
    assert forbidden.isdisjoint(MODEL_FEATURES)


def test_event_features_do_not_include_ground_truth():
    from src.event_baseline import EVENT_FEATURES

    forbidden = {"label", "anomaly_label", "incident_id", "anomaly_type"}
    assert forbidden.isdisjoint(EVENT_FEATURES)


def test_temporal_partitions_are_ordered_and_disjoint():
    config = replace(small_config(), incident_count=1)
    events, _ = generate_telemetry(config)
    windows = aggregate_windows(events, config.window_seconds)
    parts = temporal_split(windows, config.train_fraction, config.validation_fraction)

    train_times = set(parts.train["window_start"])
    validation_times = set(parts.validation["window_start"])
    test_times = set(parts.test["window_start"])
    assert train_times.isdisjoint(validation_times)
    assert train_times.isdisjoint(test_times)
    assert validation_times.isdisjoint(test_times)
    assert max(train_times) < min(validation_times) < min(test_times)


def test_incremental_predictions_match_offline_windows(tmp_path):
    config = replace(small_config(), duration_seconds=150, incident_count=1)
    events, _ = generate_telemetry(config)
    windows = aggregate_windows(events, config.window_seconds)
    parts = temporal_split(windows, config.train_fraction, config.validation_fraction)
    features = list(MODEL_FEATURES)

    pipeline = build_random_forest(config.seed)
    pipeline.set_params(model__n_estimators=5)
    pipeline.fit(parts.train[features], parts.train["anomaly_label"])
    threshold = 0.5
    artifact = tmp_path / "window_random_forest.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "threshold": threshold,
            "features": features,
            "window_seconds": config.window_seconds,
        },
        artifact,
    )

    detector = StreamingWindowDetector(artifact)
    incremental = []
    for event in events.to_dict("records"):
        incremental.extend(detector.process(event))
    incremental.extend(detector.flush())
    incremental = pd.DataFrame(incremental).sort_values(["window_start", "service"]).reset_index(drop=True)

    offline = windows.sort_values(["window_start", "service"]).reset_index(drop=True)
    offline_scores = pipeline.predict_proba(offline[features])[:, 1]
    assert len(incremental) == len(offline)
    np.testing.assert_allclose(incremental["anomaly_score"], offline_scores)
    np.testing.assert_array_equal(incremental["prediction"], (offline_scores >= threshold).astype(int))
