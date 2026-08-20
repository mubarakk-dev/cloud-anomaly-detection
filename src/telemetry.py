"""Synthetic service telemetry generation with labelled incident intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import ANOMALY_TYPES, EVENT_TYPES, SERVICES, ExperimentConfig


@dataclass(frozen=True)
class Incident:
    incident_id: str
    service: str
    anomaly_type: str
    start_offset_seconds: int
    end_offset_seconds: int
    severity: float


def generate_incidents(config: ExperimentConfig) -> list[Incident]:
    """Create service-local incidents stratified across temporal partitions."""
    config.validate()
    rng = np.random.default_rng(config.seed + 1)
    incidents: list[Incident] = []
    train_end = int(config.duration_seconds * config.train_fraction)
    validation_end = int(config.duration_seconds * (config.train_fraction + config.validation_fraction))
    partitions = (
        (0, train_end),
        (train_end, validation_end),
        (validation_end, config.duration_seconds),
    )
    # Allocate incidents approximately according to partition duration. When a
    # partition has at least four incidents, every anomaly type is represented.
    raw_counts = np.array(
        (
            config.incident_count * config.train_fraction,
            config.incident_count * config.validation_fraction,
            config.incident_count * (1 - config.train_fraction - config.validation_fraction),
        )
    )
    counts = np.floor(raw_counts).astype(int)
    for index in np.argsort(raw_counts - counts)[::-1][: config.incident_count - counts.sum()]:
        counts[index] += 1

    for (lower, upper), count in zip(partitions, counts, strict=True):
        anomaly_types = [ANOMALY_TYPES[index % len(ANOMALY_TYPES)] for index in range(count)]
        rng.shuffle(anomaly_types)
        for anomaly_type in anomaly_types:
            placed = False
            for _ in range(200):
                duration = int(rng.integers(config.incident_min_seconds, config.incident_max_seconds + 1))
                latest_start = upper - duration
                if latest_start < lower:
                    break
                start = int(rng.integers(lower, latest_start + 1))
                end = start + duration
                service = str(rng.choice(SERVICES))
                overlaps = any(
                    existing.service == service
                    and start < existing.end_offset_seconds + config.window_seconds
                    and end > existing.start_offset_seconds - config.window_seconds
                    for existing in incidents
                )
                if overlaps:
                    continue
                incidents.append(
                    Incident(
                        incident_id=f"INC-{len(incidents) + 1:03d}",
                        service=service,
                        anomaly_type=anomaly_type,
                        start_offset_seconds=start,
                        end_offset_seconds=end,
                        severity=float(rng.uniform(0.65, 1.0)),
                    )
                )
                placed = True
                break
            if not placed:
                raise RuntimeError("Could not place the requested non-overlapping incidents.")
    return sorted(incidents, key=lambda item: item.start_offset_seconds)


def _apply_incident(event: dict, incident: Incident, rng: np.random.Generator) -> None:
    severity = incident.severity
    event["incident_id"] = incident.incident_id
    event["anomaly_type"] = incident.anomaly_type
    event["label"] = 1

    if incident.anomaly_type == "latency_degradation":
        event["response_time_ms"] += rng.normal(170 + 230 * severity, 45)
        if rng.random() < 0.65:
            event["log_level"] = "WARN"
    elif incident.anomaly_type == "resource_pressure":
        event["cpu_usage"] += rng.normal(16 + 24 * severity, 5)
        event["memory_usage"] += rng.normal(10 + 20 * severity, 4)
        if rng.random() < 0.55:
            event["log_level"] = "WARN"
    elif incident.anomaly_type == "error_rate_increase":
        if rng.random() < 0.30 + 0.45 * severity:
            event["status_code"] = int(rng.choice((429, 500, 503)))
        event["log_level"] = str(rng.choice(("INFO", "WARN", "ERROR"), p=(0.20, 0.45, 0.35)))
    elif incident.anomaly_type == "partial_service_degradation":
        event["response_time_ms"] += rng.normal(100 + 170 * severity, 40)
        event["cpu_usage"] += rng.normal(8 + 14 * severity, 4)
        if rng.random() < 0.20 + 0.35 * severity:
            event["status_code"] = int(rng.choice((429, 500)))


def generate_telemetry(
    config: ExperimentConfig,
    incidents: list[Incident] | None = None,
    start_time: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one observation per service per tick and return events and incidents."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    incidents = incidents if incidents is not None else generate_incidents(config)
    start_time = start_time or datetime(2026, 1, 1, 9, 0, 0)
    incident_lookup = {(item.service, second): item for item in incidents for second in range(item.start_offset_seconds, item.end_offset_seconds)}
    rows: list[dict] = []

    service_event = dict(zip(SERVICES, EVENT_TYPES, strict=True))
    service_latency_adjustment = dict(zip(SERVICES, (20, 10, 45, 25, 65, -15), strict=True))

    for offset in range(0, config.duration_seconds, config.tick_seconds):
        timestamp = start_time + timedelta(seconds=offset)
        load_cycle = np.sin(2 * np.pi * offset / 900)
        for service in SERVICES:
            event = {
                "timestamp": timestamp,
                "service": service,
                "event_type": service_event[service],
                "response_time_ms": rng.normal(245 + service_latency_adjustment[service] + 35 * load_cycle, 65),
                "cpu_usage": rng.normal(50 + 8 * load_cycle, 10),
                "memory_usage": rng.normal(62 + 4 * load_cycle, 8),
                "status_code": int(rng.choice((200, 201, 204, 400, 404, 429, 500, 503), p=(0.68, 0.07, 0.06, 0.07, 0.05, 0.03, 0.025, 0.015))),
                "log_level": str(rng.choice(("INFO", "WARN", "ERROR"), p=(0.82, 0.13, 0.05))),
                "incident_id": None,
                "anomaly_type": "normal",
                "label": 0,
            }
            incident = incident_lookup.get((service, offset))
            if incident is not None:
                _apply_incident(event, incident, rng)

            event["response_time_ms"] = round(float(np.clip(event["response_time_ms"], 40, 2000)), 3)
            event["cpu_usage"] = round(float(np.clip(event["cpu_usage"], 1, 100)), 3)
            event["memory_usage"] = round(float(np.clip(event["memory_usage"], 5, 100)), 3)
            rows.append(event)

    incident_df = pd.DataFrame(asdict(item) for item in incidents)
    return pd.DataFrame(rows), incident_df
