"""Configuration for reproducible synthetic cloud telemetry experiments."""

from dataclasses import dataclass


SERVICES = (
    "api-gateway",
    "auth-service",
    "payment-service",
    "inventory-service",
    "database-service",
    "cache-service",
)

EVENT_TYPES = (
    "request",
    "login",
    "payment",
    "inventory_check",
    "db_query",
    "cache_lookup",
)

ANOMALY_TYPES = (
    "latency_degradation",
    "resource_pressure",
    "error_rate_increase",
    "partial_service_degradation",
)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    duration_seconds: int = 3600
    tick_seconds: int = 1
    window_seconds: int = 30
    incident_count: int = 24
    incident_min_seconds: int = 30
    incident_max_seconds: int = 90
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    rf_threshold: float = 0.40
    isolation_contamination: float = 0.05

    def validate(self) -> None:
        if self.duration_seconds <= 0 or self.tick_seconds <= 0:
            raise ValueError("Durations must be positive.")
        if self.duration_seconds % self.tick_seconds:
            raise ValueError("duration_seconds must be divisible by tick_seconds.")
        if self.window_seconds % self.tick_seconds:
            raise ValueError("window_seconds must be divisible by tick_seconds.")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between zero and one.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one.")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("A non-empty test partition is required.")
