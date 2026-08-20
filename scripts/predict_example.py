"""Run one finite raw-event replay directly through the persisted detector."""

from pathlib import Path

from src.config import ExperimentConfig
from src.stream_detector import StreamingWindowDetector
from src.telemetry import generate_telemetry


def main() -> None:
    detector = StreamingWindowDetector(Path("models/window_random_forest.joblib"))
    events, _ = generate_telemetry(
        ExperimentConfig(
            seed=2026,
            duration_seconds=120,
            window_seconds=detector.window_seconds,
            incident_count=1,
            incident_min_seconds=20,
            incident_max_seconds=40,
        )
    )
    predictions = []
    for event in events.to_dict("records"):
        predictions.extend(detector.process(event))
    predictions.extend(detector.flush())
    for result in predictions:
        if result["prediction"]:
            print(
                f'{result["window_end"]} | {result["service"]:<20} | '
                f'ANOMALOUS score={result["anomaly_score"]:.3f} threshold={result["threshold"]:.3f}'
            )


if __name__ == "__main__":
    main()
