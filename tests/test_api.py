from fastapi.testclient import TestClient

from app.api import app
from src.config import ExperimentConfig
from src.telemetry import generate_telemetry


client = TestClient(app)


def public_event(row: dict) -> dict:
    fields = (
        "timestamp", "service", "event_type", "response_time_ms", "cpu_usage",
        "memory_usage", "status_code", "log_level",
    )
    result = {field: row[field] for field in fields}
    result["timestamp"] = result["timestamp"].isoformat()
    return result


def test_health_exposes_artifact_configuration():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["window_seconds"] == 30
    assert 0 <= response.json()["threshold"] <= 1


def test_api_accepts_raw_events_and_never_returns_ground_truth():
    client.post("/reset")
    events, _ = generate_telemetry(
        ExperimentConfig(seed=11, duration_seconds=60, window_seconds=30, incident_count=0)
    )
    predictions = []
    for row in events.to_dict("records"):
        response = client.post("/events", json=public_event(row))
        assert response.status_code == 200
        predictions.extend(response.json())
    predictions.extend(client.post("/flush").json())
    assert predictions
    assert {"ground_truth", "incident_id", "anomaly_type"}.isdisjoint(predictions[0])


def test_api_schema_excludes_ground_truth_fields():
    schema = client.get("/openapi.json").json()["components"]["schemas"]["RawTelemetryEvent"]
    assert {"label", "incident_id", "anomaly_type"}.isdisjoint(schema["properties"])
