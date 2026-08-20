"""Train and persist the same leakage-aware pipeline used by the research study."""

from pathlib import Path

from src.config import ExperimentConfig
from src.run_experiment import run


def main() -> None:
    metrics = run(
        ExperimentConfig(seed=42, duration_seconds=3600, window_seconds=30, incident_count=24),
        Path("outputs"),
    )
    print(metrics.to_string(index=False))
    print("\nArtifacts written to models/window_random_forest.joblib and models/window_isolation_forest.joblib")


if __name__ == "__main__":
    main()
