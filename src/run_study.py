"""Run repeated experiments across seeds and window sizes for robust reporting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .event_baseline import run_event_baseline
from .run_experiment import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2026])
    parser.add_argument("--window-seconds", type=int, nargs="+", default=[15, 30, 60])
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--incident-count", type=int, default=24)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/study"))
    args = parser.parse_args()

    collected: list[pd.DataFrame] = []
    event_collected: list[pd.DataFrame] = []
    # Event baselines do not depend on window length and are run once per seed.
    for seed in args.seeds:
        config = ExperimentConfig(seed=seed, duration_seconds=args.duration_seconds, incident_count=args.incident_count)
        metrics = run_event_baseline(config, args.output_root / "event" / f"seed_{seed}")
        metrics.insert(0, "seed", seed)
        metrics.insert(1, "representation", "individual_event")
        event_collected.append(metrics)
        print(f"Completed event baseline seed={seed}")

    for window_seconds in args.window_seconds:
        for seed in args.seeds:
            run_root = args.output_root / f"window_{window_seconds}s" / f"seed_{seed}"
            config = ExperimentConfig(
                seed=seed,
                duration_seconds=args.duration_seconds,
                window_seconds=window_seconds,
                incident_count=args.incident_count,
            )
            metrics = run(config, run_root)
            metrics.insert(0, "seed", seed)
            metrics.insert(1, "window_seconds", window_seconds)
            metrics.insert(2, "representation", "service_window")
            collected.append(metrics)
            print(f"Completed seed={seed}, window={window_seconds}s")

    results = pd.concat(collected, ignore_index=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_root / "all_runs.csv", index=False)
    event_results = pd.concat(event_collected, ignore_index=True)
    event_results.to_csv(args.output_root / "event_all_runs.csv", index=False)
    summary = results.groupby(["window_seconds", "model"]).agg(
        runs=("seed", "count"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        fpr_mean=("false_positive_rate", "mean"),
        fpr_std=("false_positive_rate", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        incident_recall_mean=("incident_recall", "mean"),
        detection_delay_mean=("mean_detection_delay_seconds", "mean"),
    ).reset_index()
    summary.to_csv(args.output_root / "summary.csv", index=False)
    event_summary = event_results.groupby(["representation", "model"]).agg(
        runs=("seed", "count"),
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        fpr_mean=("false_positive_rate", "mean"), fpr_std=("false_positive_rate", "std"),
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        incident_recall_mean=("incident_recall", "mean"),
        detection_delay_mean=("mean_detection_delay_seconds", "mean"),
    ).reset_index()
    event_summary.to_csv(args.output_root / "event_summary.csv", index=False)
    print("\n", summary.to_string(index=False))
    print("\nEvent baseline\n", event_summary.to_string(index=False))


if __name__ == "__main__":
    main()
