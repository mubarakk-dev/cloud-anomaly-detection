"""Recalculate aggregate metrics that depend only on saved predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .modelling import incident_metrics


def _refresh_file(metrics_path: Path, predictions_path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path, parse_dates=["window_start", "window_end"])
    for index, row in metrics.iterrows():
        model_predictions = predictions[predictions["model"] == row["model"]]
        updated = incident_metrics(model_predictions)
        for key, value in updated.items():
            metrics.loc[index, key] = value
    metrics.to_csv(metrics_path, index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=Path("outputs/study_v2"))
    args = parser.parse_args()
    window_frames = []
    for metrics_path in args.study_root.glob("window_*s/seed_*/metrics/model_comparison.csv"):
        run_root = metrics_path.parents[1]
        metrics = _refresh_file(metrics_path, run_root / "predictions" / "test_predictions.csv")
        window_seconds = int(metrics_path.parents[2].name.removeprefix("window_").removesuffix("s"))
        seed = int(run_root.name.removeprefix("seed_"))
        metrics.insert(0, "seed", seed)
        metrics.insert(1, "window_seconds", window_seconds)
        metrics.insert(2, "representation", "service_window")
        window_frames.append(metrics)
    all_runs = pd.concat(window_frames, ignore_index=True).sort_values(["window_seconds", "seed", "model"])
    all_runs.to_csv(args.study_root / "all_runs.csv", index=False)
    summary = all_runs.groupby(["window_seconds", "model"]).agg(
        runs=("seed", "count"), precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"), f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        fpr_mean=("false_positive_rate", "mean"), fpr_std=("false_positive_rate", "std"),
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        incident_recall_mean=("incident_recall", "mean"), detection_delay_mean=("mean_detection_delay_seconds", "mean"),
    ).reset_index()
    summary.to_csv(args.study_root / "summary.csv", index=False)

    event_frames = []
    for metrics_path in args.study_root.glob("event/seed_*/event_metrics.csv"):
        run_root = metrics_path.parent
        metrics = _refresh_file(metrics_path, run_root / "event_predictions.csv")
        metrics.insert(0, "seed", int(run_root.name.removeprefix("seed_")))
        metrics.insert(1, "representation", "individual_event")
        event_frames.append(metrics)
    event_runs = pd.concat(event_frames, ignore_index=True).sort_values(["seed", "model"])
    event_runs.to_csv(args.study_root / "event_all_runs.csv", index=False)
    event_summary = event_runs.groupby(["representation", "model"]).agg(
        runs=("seed", "count"), precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"), f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        fpr_mean=("false_positive_rate", "mean"), fpr_std=("false_positive_rate", "std"),
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        incident_recall_mean=("incident_recall", "mean"), detection_delay_mean=("mean_detection_delay_seconds", "mean"),
    ).reset_index()
    event_summary.to_csv(args.study_root / "event_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nEvent baseline\n", event_summary.to_string(index=False))


if __name__ == "__main__":
    main()
