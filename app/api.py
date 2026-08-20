"""FastAPI interface for the near-real-time incremental inference prototype."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.stream_detector import StreamingWindowDetector


ARTIFACT_PATH = Path(os.getenv("MODEL_ARTIFACT", "models/window_random_forest.joblib"))
_detector: StreamingWindowDetector | None = None
_lock = Lock()


class RawTelemetryEvent(BaseModel):
    """Observable fields accepted by the API; ground truth is intentionally absent."""

    timestamp: datetime
    service: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    response_time_ms: float = Field(ge=0)
    cpu_usage: float = Field(ge=0, le=100)
    memory_usage: float = Field(ge=0, le=100)
    status_code: int = Field(ge=100, le=599)
    log_level: str = Field(pattern="^(INFO|WARN|ERROR)$")


class Prediction(BaseModel):
    window_start: datetime
    window_end: datetime
    service: str
    prediction: int
    label: str
    anomaly_score: float
    threshold: float
    inference_time_ms: float


def _load_detector() -> StreamingWindowDetector:
    global _detector
    if _detector is None:
        if not ARTIFACT_PATH.exists():
            raise RuntimeError(
                f"Model artifact not found at {ARTIFACT_PATH}. "
                "Run `python -m training.train_window_model` first."
            )
        _detector = StreamingWindowDetector(ARTIFACT_PATH)
    return _detector


app = FastAPI(
    title="Cloud Telemetry Anomaly Detection",
    description=(
        "Research prototype that buffers raw service telemetry, calculates the "
        "validated behavioural-window representation and applies a persisted model."
    ),
    version="2.0.0",
)


@app.get("/")
def home() -> dict:
    return {"name": app.title, "documentation": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict:
    detector = _load_detector()
    return {
        "status": "ok",
        "model": "random_forest",
        "window_seconds": detector.window_seconds,
        "threshold": detector.threshold,
        "prototype": True,
    }


@app.post("/events", response_model=list[Prediction])
def ingest_event(data: RawTelemetryEvent) -> list[Prediction]:
    """Accept one raw observation and return any windows closed by its timestamp."""
    event = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    # The shared aggregator expects audit metadata. Neutral values are internal
    # only and are excluded from MODEL_FEATURES and the API response.
    event.update({"label": 0, "incident_id": None, "anomaly_type": "normal"})
    try:
        with _lock:
            results = _load_detector().process(event)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [_public_prediction(item) for item in results]


@app.post("/flush", response_model=list[Prediction])
def flush() -> list[Prediction]:
    """Close buffered windows; intended for finite demonstrations and tests."""
    with _lock:
        results = _load_detector().flush()
    return [_public_prediction(item) for item in results]


@app.post("/reset")
def reset() -> dict[str, str]:
    """Clear in-memory buffers without changing the persisted model."""
    with _lock:
        _load_detector().buffers.clear()
    return {"status": "reset"}


def _public_prediction(result: dict) -> dict:
    """Remove simulator-only ground truth from the operational response."""
    return {
        "window_start": result["window_start"],
        "window_end": result["window_end"],
        "service": result["service"],
        "prediction": result["prediction"],
        "label": "ANOMALOUS" if result["prediction"] else "NORMAL",
        "anomaly_score": result["anomaly_score"],
        "threshold": result["threshold"],
        "inference_time_ms": result["inference_time_ms"],
    }
