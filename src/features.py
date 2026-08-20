"""Batch and incremental feature extraction for service telemetry windows."""

from __future__ import annotations

import pandas as pd


MODEL_FEATURES = (
    "service",
    "avg_response_time",
    "max_response_time",
    "avg_cpu_usage",
    "max_cpu_usage",
    "avg_memory_usage",
    "max_memory_usage",
    "error_rate",
    "warn_rate",
    "error_log_rate",
    "log_count",
)


def aggregate_windows(events: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    """Aggregate raw telemetry into fixed service windows."""
    required = {
        "timestamp", "service", "response_time_ms", "cpu_usage", "memory_usage",
        "status_code", "log_level", "label", "incident_id", "anomaly_type",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing telemetry columns: {sorted(missing)}")

    frame = events.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["window_start"] = frame["timestamp"].dt.floor(f"{window_seconds}s")
    windows = frame.groupby(["window_start", "service"], sort=True).agg(
        avg_response_time=("response_time_ms", "mean"),
        max_response_time=("response_time_ms", "max"),
        avg_cpu_usage=("cpu_usage", "mean"),
        max_cpu_usage=("cpu_usage", "max"),
        avg_memory_usage=("memory_usage", "mean"),
        max_memory_usage=("memory_usage", "max"),
        error_rate=("status_code", lambda values: (values >= 500).mean()),
        warn_rate=("log_level", lambda values: (values == "WARN").mean()),
        error_log_rate=("log_level", lambda values: (values == "ERROR").mean()),
        log_count=("timestamp", "count"),
        anomaly_label=("label", "max"),
        incident_id=("incident_id", lambda values: next((value for value in values if pd.notna(value)), None)),
        anomaly_type=("anomaly_type", lambda values: next((value for value in values if value != "normal"), "normal")),
    ).reset_index()
    windows["window_end"] = windows["window_start"] + pd.to_timedelta(window_seconds, unit="s")
    return windows
