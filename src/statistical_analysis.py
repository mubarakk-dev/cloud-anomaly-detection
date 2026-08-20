"""Paired non-parametric comparisons for repeated-seed experiment results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired_result(name: str, first: pd.Series, second: pd.Series) -> dict:
    differences = first.to_numpy() - second.to_numpy()
    if np.allclose(differences, 0):
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = wilcoxon(first, second, alternative="two-sided", method="exact")
    return {
        "comparison": name,
        "pairs": len(first),
        "first_mean": float(first.mean()),
        "second_mean": float(second.mean()),
        "mean_paired_difference": float(differences.mean()),
        "median_paired_difference": float(np.median(differences)),
        "wilcoxon_statistic": float(statistic),
        "p_value_two_sided": float(p_value),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=Path("outputs/study_v2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/report_materials_v2/tables/paired_tests.csv"))
    args = parser.parse_args()
    windows = pd.read_csv(args.study_root / "all_runs.csv")
    events = pd.read_csv(args.study_root / "event_all_runs.csv")
    rows = []

    for model in ("random_forest", "isolation_forest"):
        event = events[events["model"] == model].set_index("seed")["f1"]
        for duration in (15, 30, 60):
            window = windows[(windows["model"] == model) & (windows["window_seconds"] == duration)].set_index("seed")["f1"]
            rows.append(paired_result(f"{model}: {duration}s window minus individual event", window.loc[event.index], event))

    for duration in (15, 30, 60):
        rf = windows[(windows["model"] == "random_forest") & (windows["window_seconds"] == duration)].set_index("seed")["f1"]
        iso = windows[(windows["model"] == "isolation_forest") & (windows["window_seconds"] == duration)].set_index("seed")["f1"]
        rows.append(paired_result(f"{duration}s: Random Forest minus Isolation Forest", rf.loc[iso.index], iso))

    rf = windows[windows["model"] == "random_forest"].pivot(index="seed", columns="window_seconds", values="f1")
    rows.append(paired_result("Random Forest: 60s minus 15s", rf[60], rf[15]))
    rows.append(paired_result("Random Forest: 30s minus 15s", rf[30], rf[15]))

    results = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
