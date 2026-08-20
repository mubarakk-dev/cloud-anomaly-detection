"""Streamlit dashboard for the accelerated incremental inference demonstration."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import ExperimentConfig
from src.stream_detector import StreamingWindowDetector
from src.telemetry import generate_telemetry


ARTIFACT = Path("models/window_random_forest.joblib")

st.set_page_config(page_title="Cloud Anomaly Detection", page_icon="📈", layout="wide")
st.title("Cloud Telemetry Anomaly Detection")
st.caption("Accelerated local research prototype · behavioural service windows · validated Random Forest threshold")

with st.sidebar:
    st.header("Simulation")
    duration = st.slider("Simulated duration (seconds)", 300, 900, 600, 30)
    seed = st.number_input("Random seed", min_value=1, value=2026, step=1)
    show_truth = st.toggle("Show simulator ground truth", value=True)
    run_demo = st.button("Run accelerated replay", type="primary", use_container_width=True)

if not ARTIFACT.exists():
    st.error("Model artifact is missing. Run `python -m training.train_window_model`.")
    st.stop()

preview = StreamingWindowDetector(ARTIFACT)
c1, c2, c3 = st.columns(3)
c1.metric("Model", "Random Forest")
c2.metric("Window duration", f"{preview.window_seconds} s")
c3.metric("Validated threshold", f"{preview.threshold:.3f}")

st.info(
    "The replay sends raw observations to per-service buffers. A prediction becomes "
    "available only after a window closes. Ground truth is retained solely to evaluate the simulation."
)

if run_demo:
    config = ExperimentConfig(
        seed=int(seed),
        duration_seconds=int(duration),
        window_seconds=preview.window_seconds,
        incident_count=max(3, int(duration) // 150),
        incident_min_seconds=30,
        incident_max_seconds=60,
    )
    events, _ = generate_telemetry(config)
    detector = StreamingWindowDetector(ARTIFACT)
    predictions: list[dict] = []
    progress = st.progress(0, text="Processing raw telemetry")
    ticks = list(events.groupby("timestamp", sort=True))
    for index, (_, tick) in enumerate(ticks, start=1):
        for event in tick.to_dict("records"):
            predictions.extend(detector.process(event))
        if index % 20 == 0 or index == len(ticks):
            progress.progress(index / len(ticks), text=f"Processed {index:,} of {len(ticks):,} simulated seconds")
    predictions.extend(detector.flush())
    frame = pd.DataFrame(predictions).sort_values(["window_end", "service"])
    frame["status"] = frame["prediction"].map({0: "NORMAL", 1: "ANOMALOUS"})

    anomalous = int(frame["prediction"].sum())
    detected_incidents = frame.loc[frame["prediction"].eq(1), "incident_id"].dropna().nunique()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raw observations", f"{len(events):,}")
    m2.metric("Completed windows", f"{len(frame):,}")
    m3.metric("Anomaly alerts", anomalous)
    m4.metric("Detected simulated incidents", int(detected_incidents))

    chart = frame.pivot_table(index="window_end", columns="service", values="anomaly_score", aggfunc="mean")
    st.subheader("Anomaly score over simulated time")
    st.line_chart(chart)

    display_columns = ["window_end", "service", "status", "anomaly_score", "threshold", "inference_time_ms"]
    if show_truth:
        display_columns += ["ground_truth", "anomaly_type", "incident_id"]
    st.subheader("Completed service windows")
    st.dataframe(frame[display_columns].sort_values("window_end", ascending=False), use_container_width=True, hide_index=True)
    st.download_button(
        "Download predictions",
        frame.to_csv(index=False).encode("utf-8"),
        "incremental_predictions.csv",
        "text/csv",
    )
else:
    st.subheader("What this demonstrates")
    st.markdown(
        """
        1. Raw synthetic observations arrive in timestamp order.
        2. State is maintained separately for each service and fixed time window.
        3. Completed windows use the same feature function as offline evaluation.
        4. The persisted preprocessing/model pipeline and validation-selected threshold produce each alert.

        This is an accelerated local simulation, not a production-scale streaming benchmark.
        """
    )
