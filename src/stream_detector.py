"""Accelerated near-real-time telemetry generation and window inference demo."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd

from .config import ExperimentConfig
from .features import aggregate_windows
from .telemetry import generate_telemetry


class StreamingWindowDetector:
    """Buffer raw observations and classify each completed service window."""

    def __init__(self, artifact_path: Path):
        artifact = joblib.load(artifact_path)
        self.pipeline = artifact["pipeline"]
        self.threshold = float(artifact["threshold"])
        self.features = list(artifact["features"])
        self.window_seconds = int(artifact["window_seconds"])
        self.buffers: dict[tuple[pd.Timestamp, str], list[dict]] = defaultdict(list)

    def process(self, event: dict) -> list[dict]:
        timestamp = pd.Timestamp(event["timestamp"])
        window_start = timestamp.floor(f"{self.window_seconds}s")
        key = (window_start, str(event["service"]))
        self.buffers[key].append(event)
        return self._close_before(timestamp)

    def flush(self) -> list[dict]:
        results: list[dict] = []
        for key in sorted(self.buffers):
            results.append(self._classify(key, self.buffers[key]))
        self.buffers.clear()
        return results

    def _close_before(self, timestamp: pd.Timestamp) -> list[dict]:
        duration = pd.Timedelta(self.window_seconds, unit="s")
        completed = [key for key in self.buffers if key[0] + duration <= timestamp]
        results = [self._classify(key, self.buffers.pop(key)) for key in sorted(completed)]
        return results

    def _classify(self, key: tuple[pd.Timestamp, str], events: list[dict]) -> dict:
        started = time.perf_counter()
        window = aggregate_windows(pd.DataFrame(events), self.window_seconds).iloc[0]
        score = float(self.pipeline.predict_proba(pd.DataFrame([window[self.features]]))[:, 1][0])
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "window_start": key[0],
            "window_end": key[0] + pd.Timedelta(self.window_seconds, unit="s"),
            "service": key[1],
            "prediction": int(score >= self.threshold),
            "anomaly_score": score,
            "threshold": self.threshold,
            "inference_time_ms": latency_ms,
            "ground_truth": int(window["anomaly_label"]),
            "incident_id": window["incident_id"],
            "anomaly_type": window["anomaly_type"],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=Path("models/window_random_forest.joblib"))
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--incident-count", type=int, default=4)
    parser.add_argument("--speed", type=float, default=0.01, help="Real seconds to wait after each simulated tick.")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    detector = StreamingWindowDetector(args.artifact)
    config = ExperimentConfig(
        seed=args.seed,
        duration_seconds=args.duration_seconds,
        window_seconds=detector.window_seconds,
        incident_count=args.incident_count,
    )
    events, _ = generate_telemetry(config)
    print("Streaming synthetic telemetry. Ground truth is shown for demonstration only.\n")
    previous_timestamp = None
    for timestamp, tick in events.groupby("timestamp", sort=True):
        for event in tick.to_dict("records"):
            for result in detector.process(event):
                status = "ANOMALOUS" if result["prediction"] else "NORMAL"
                truth = "ANOMALOUS" if result["ground_truth"] else "NORMAL"
                print(
                    f'{result["window_end"]} | {result["service"]:<20} | '
                    f'prediction={status:<9} score={result["anomaly_score"]:.3f} '
                    f'truth={truth:<9} latency={result["inference_time_ms"]:.2f}ms'
                )
        if previous_timestamp is not None and args.speed > 0:
            time.sleep(args.speed)
        previous_timestamp = timestamp
    for result in detector.flush():
        status = "ANOMALOUS" if result["prediction"] else "NORMAL"
        truth = "ANOMALOUS" if result["ground_truth"] else "NORMAL"
        print(f'{result["window_end"]} | {result["service"]:<20} | prediction={status:<9} score={result["anomaly_score"]:.3f} truth={truth}')


if __name__ == "__main__":
    main()
