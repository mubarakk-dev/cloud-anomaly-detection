"""Replay raw synthetic telemetry into the HTTP ingestion endpoint."""

import argparse
import time

import requests

from src.config import ExperimentConfig
from src.telemetry import generate_telemetry


MODEL_INPUT_FIELDS = (
    "timestamp", "service", "event_type", "response_time_ms", "cpu_usage",
    "memory_usage", "status_code", "log_level",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--speed", type=float, default=0.02, help="Wall-clock pause per simulated second.")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.duration_seconds < 150:
        parser.error("--duration-seconds must be at least 150 so every temporal partition can contain an incident.")

    requests.post(f"{args.api_url}/reset", timeout=10).raise_for_status()
    events, _ = generate_telemetry(
        ExperimentConfig(
            seed=args.seed,
            duration_seconds=args.duration_seconds,
            incident_count=max(3, args.duration_seconds // 150),
            incident_min_seconds=30,
            incident_max_seconds=min(60, args.duration_seconds // 5),
        )
    )
    print(f"Replaying {len(events):,} raw observations to {args.api_url}/events")
    for _, tick in events.groupby("timestamp", sort=True):
        for row in tick.to_dict("records"):
            payload = {field: row[field] for field in MODEL_INPUT_FIELDS}
            payload["timestamp"] = payload["timestamp"].isoformat()
            response = requests.post(f"{args.api_url}/events", json=payload, timeout=10)
            response.raise_for_status()
            for result in response.json():
                print(
                    f'{result["window_end"]} | {result["service"]:<20} | '
                    f'{result["label"]:<9} score={result["anomaly_score"]:.3f} '
                    f'latency={result["inference_time_ms"]:.2f}ms'
                )
        if args.speed > 0:
            time.sleep(args.speed)
    response = requests.post(f"{args.api_url}/flush", timeout=30)
    response.raise_for_status()
    for result in response.json():
        print(f'{result["window_end"]} | {result["service"]:<20} | {result["label"]}')


if __name__ == "__main__":
    main()
