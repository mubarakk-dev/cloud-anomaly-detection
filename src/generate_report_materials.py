"""Create dissertation-ready tables and figures from repeated-study outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {"random_forest": "Random Forest", "isolation_forest": "Isolation Forest"}


def _save_metric_chart(summary: pd.DataFrame, metric: str, ylabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    windows = sorted(summary["window_seconds"].unique())
    positions = np.arange(len(windows))
    width = 0.34
    for offset, model in zip((-width / 2, width / 2), ("random_forest", "isolation_forest"), strict=True):
        rows = summary[summary["model"] == model].set_index("window_seconds").loc[windows]
        ax.bar(
            positions + offset,
            rows[f"{metric}_mean"],
            width,
            yerr=rows[f"{metric}_std"],
            capsize=4,
            label=MODEL_LABELS[model],
        )
    ax.set_xticks(positions, [f"{value} s" for value in windows])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Service-window duration")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=Path("outputs/study"))
    parser.add_argument("--report-root", type=Path, default=Path("outputs/report_materials"))
    args = parser.parse_args()
    figures = args.report_root / "figures"
    tables = args.report_root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    all_runs = pd.read_csv(args.study_root / "all_runs.csv")
    summary = pd.read_csv(args.study_root / "summary.csv")
    event_summary = pd.read_csv(args.study_root / "event_summary.csv")
    _save_metric_chart(summary, "f1", "Anomaly F1-score (mean ± SD)", figures / "f1_by_window.png")
    _save_metric_chart(summary, "recall", "Anomaly recall (mean ± SD)", figures / "recall_by_window.png")
    _save_metric_chart(summary, "pr_auc", "PR-AUC (mean ± SD)", figures / "pr_auc_by_window.png")

    compact = summary[[
        "window_seconds", "model", "precision_mean", "precision_std", "recall_mean", "recall_std",
        "f1_mean", "f1_std", "fpr_mean", "fpr_std", "pr_auc_mean", "pr_auc_std",
        "incident_recall_mean", "detection_delay_mean",
    ]].copy()
    compact["model"] = compact["model"].map(MODEL_LABELS)
    compact.to_csv(tables / "model_window_summary.csv", index=False)
    event_compact = event_summary.copy()
    event_compact["model"] = event_compact["model"].map(MODEL_LABELS)
    event_compact.to_csv(tables / "event_baseline_summary.csv", index=False)

    # Direct representation comparison: individual events versus all windows.
    comparison_rows = []
    for _, row in event_summary.iterrows():
        comparison_rows.append({"representation": "Individual events", "model": row["model"], "f1_mean": row["f1_mean"], "f1_std": row["f1_std"]})
    for _, row in summary.iterrows():
        comparison_rows.append({"representation": f'{int(row["window_seconds"])} s windows', "model": row["model"], "f1_mean": row["f1_mean"], "f1_std": row["f1_std"]})
    comparison = pd.DataFrame(comparison_rows)
    labels = ["Individual events", "15 s windows", "30 s windows", "60 s windows"]
    positions = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, model in zip((-width / 2, width / 2), ("random_forest", "isolation_forest"), strict=True):
        rows = comparison[comparison["model"] == model].set_index("representation").loc[labels]
        ax.bar(positions + offset, rows["f1_mean"], width, yerr=rows["f1_std"], capsize=4, label=MODEL_LABELS[model])
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Anomaly F1-score (mean ± SD)")
    ax.set_xlabel("Telemetry representation")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "event_vs_window_f1.png", dpi=300)
    plt.close(fig)

    tradeoff = compact[compact["model"] == "Random Forest"].sort_values("window_seconds")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.errorbar(tradeoff["window_seconds"], tradeoff["f1_mean"], yerr=tradeoff["f1_std"], marker="o", capsize=4, label="F1-score")
    ax1.set_xlabel("Service-window duration (seconds)")
    ax1.set_ylabel("Anomaly F1-score")
    # Use the full metric range so the small F1 differences are not visually exaggerated.
    ax1.set_ylim(0.0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(tradeoff["window_seconds"], tradeoff["detection_delay_mean"], marker="s", color="tab:red", label="Detection delay")
    ax2.set_ylabel("Mean detection delay (seconds)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0.0, float(tradeoff["detection_delay_mean"].max()) * 1.15)
    ax1.grid(alpha=0.25)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(figures / "window_accuracy_latency_tradeoff.png", dpi=300)
    plt.close(fig)

    per_type_rows = []
    for window_seconds in sorted(all_runs["window_seconds"].unique()):
        for seed in sorted(all_runs["seed"].unique()):
            prediction_path = args.study_root / f"window_{window_seconds}s" / f"seed_{seed}" / "predictions" / "test_predictions.csv"
            predictions = pd.read_csv(prediction_path)
            anomalous = predictions[predictions["anomaly_label"] == 1]
            for (model, anomaly_type), rows in anomalous.groupby(["model", "anomaly_type"]):
                per_type_rows.append({
                    "window_seconds": window_seconds,
                    "seed": seed,
                    "model": model,
                    "anomaly_type": anomaly_type,
                    "anomalous_windows": len(rows),
                    "detected_windows": int(rows["prediction"].sum()),
                    "recall": float(rows["prediction"].mean()),
                })
    per_type = pd.DataFrame(per_type_rows)
    per_type.to_csv(tables / "per_anomaly_all_runs.csv", index=False)
    per_type_summary = per_type.groupby(["window_seconds", "model", "anomaly_type"]).agg(
        runs_present=("seed", "count"),
        anomalous_windows=("anomalous_windows", "sum"),
        detected_windows=("detected_windows", "sum"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
    ).reset_index()
    per_type_summary["model"] = per_type_summary["model"].map(MODEL_LABELS)
    per_type_summary.to_csv(tables / "per_anomaly_summary.csv", index=False)

    best = compact.sort_values("f1_mean", ascending=False).iloc[0]
    event_rf = event_compact[event_compact["model"] == "Random Forest"].iloc[0]
    event_iso = event_compact[event_compact["model"] == "Isolation Forest"].iloc[0]
    lines = [
        "# Verified Experimental Results",
        "",
        "These results were generated from five independent seeds (42, 123, 456, 789, and 2026) using chronological 60/20/20 train-validation-test partitions.",
        "",
        "## Principal result",
        "",
        f'The highest mean anomaly F1-score was obtained by **{best["model"]} with {int(best["window_seconds"])}-second windows**: '
        f'F1 = {best["f1_mean"]:.3f} ± {best["f1_std"]:.3f}, recall = {best["recall_mean"]:.3f} ± {best["recall_std"]:.3f}, '
        f'precision = {best["precision_mean"]:.3f} ± {best["precision_std"]:.3f}, and false-positive rate = {best["fpr_mean"]:.3f} ± {best["fpr_std"]:.3f}.',
        "",
        "## Effect of behavioural aggregation",
        "",
        f'Individual-event Random Forest achieved F1 = {event_rf["f1_mean"]:.3f} ± {event_rf["f1_std"]:.3f}, while individual-event Isolation Forest achieved F1 = {event_iso["f1_mean"]:.3f} ± {event_iso["f1_std"]:.3f}. All tested window representations improved mean F1 for both models.',
        "",
        "The highest F1 does not automatically define the best operational window. Longer windows accumulate more behavioural evidence but delay prediction availability. The 15-second Random Forest was close to the 60-second model in F1 while producing substantially lower mean detection delay and false-positive rate; this trade-off should be discussed rather than presenting one universally optimal duration.",
        "",
        "## Interpretation constraints",
        "",
        "- Results apply to the implemented synthetic environment and do not establish production-cloud generalisability.",
        "- Incident recall was 1.0 for every Random Forest configuration and most Isolation Forest configurations. This only means that at least one associated window was detected; window recall was lower.",
        "- Standard deviations show material seed sensitivity, especially for 60-second windows and Isolation Forest.",
        "- Thresholds were selected independently on each validation partition and never tuned on its test partition.",
        "- The strong scores require further discussion as evidence that the simulated incidents may remain easier to identify than real incidents.",
        "",
        "## Generated evidence",
        "",
        "- `tables/model_window_summary.csv`: full model/window comparison.",
        "- `tables/per_anomaly_summary.csv`: anomaly-type recall.",
        "- `figures/f1_by_window.png`: mean F1 with standard deviation.",
        "- `figures/recall_by_window.png`: mean recall with standard deviation.",
        "- `figures/pr_auc_by_window.png`: mean PR-AUC with standard deviation.",
        "- `figures/event_vs_window_f1.png`: direct representation comparison.",
        "- `figures/window_accuracy_latency_tradeoff.png`: accuracy-delay trade-off.",
    ]
    (args.report_root / "RESULTS_NOTES.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
