import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# 1. Generate simulated cloud logs
np.random.seed(42)

n_logs = 10000

services = [
    "api-gateway",
    "auth-service",
    "payment-service",
    "inventory-service",
    "database-service",
    "cache-service"
]

event_types = [
    "request",
    "login",
    "db_query",
    "cache_lookup",
    "payment",
    "inventory_check"
]

start_time = datetime(2026, 1, 1, 9, 0, 0)

cloud_df = pd.DataFrame({
    "timestamp": [start_time + timedelta(seconds=i) for i in range(n_logs)],
    "service": np.random.choice(services, n_logs),
    "event_type": np.random.choice(event_types, n_logs),
    "response_time_ms": np.random.normal(260, 110, n_logs),
    "cpu_usage": np.random.normal(55, 16, n_logs),
    "memory_usage": np.random.normal(64, 12, n_logs),
    "status_code": np.random.choice(
        [200, 201, 204, 400, 404, 429, 500, 503],
        n_logs,
        p=[0.63, 0.07, 0.06, 0.08, 0.06, 0.04, 0.04, 0.02]
    ),
    "log_level": np.random.choice(
        ["INFO", "WARN", "ERROR"],
        n_logs,
        p=[0.78, 0.16, 0.06]
    ),
    "label": 0,
    "anomaly_type": "normal"
})

cloud_df["response_time_ms"] = cloud_df["response_time_ms"].clip(80, 900)
cloud_df["cpu_usage"] = cloud_df["cpu_usage"].clip(10, 98)
cloud_df["memory_usage"] = cloud_df["memory_usage"].clip(25, 98)


# 2. Inject contextual anomalies
anomaly_windows = np.random.choice(
    np.arange(100, n_logs - 100),
    size=25,
    replace=False
)

for start_idx in anomaly_windows:
    window_size = np.random.randint(10, 25)
    affected_indices = np.arange(start_idx, start_idx + window_size)

    affected_service = np.random.choice(services)

    anomaly_type = np.random.choice([
        "latency_degradation",
        "resource_pressure",
        "error_rate_increase",
        "partial_service_degradation"
    ])

    service_mask = cloud_df.loc[affected_indices, "service"] == affected_service
    selected_indices = affected_indices[service_mask.values]

    if len(selected_indices) < 5:
        selected_indices = affected_indices[:np.random.randint(5, min(15, len(affected_indices)))]

    cloud_df.loc[selected_indices, "label"] = 1
    cloud_df.loc[selected_indices, "anomaly_type"] = anomaly_type

    if anomaly_type == "latency_degradation":
        cloud_df.loc[selected_indices, "response_time_ms"] += np.random.normal(
            220, 80, len(selected_indices)
        )
        cloud_df.loc[selected_indices, "log_level"] = np.random.choice(
            ["INFO", "WARN"],
            len(selected_indices),
            p=[0.45, 0.55]
        )

    elif anomaly_type == "resource_pressure":
        cloud_df.loc[selected_indices, "cpu_usage"] += np.random.normal(
            18, 8, len(selected_indices)
        )
        cloud_df.loc[selected_indices, "memory_usage"] += np.random.normal(
            12, 6, len(selected_indices)
        )
        cloud_df.loc[selected_indices, "log_level"] = np.random.choice(
            ["INFO", "WARN"],
            len(selected_indices),
            p=[0.5, 0.5]
        )

    elif anomaly_type == "error_rate_increase":
        cloud_df.loc[selected_indices, "status_code"] = np.random.choice(
            [200, 429, 500, 503],
            len(selected_indices),
            p=[0.35, 0.25, 0.25, 0.15]
        )
        cloud_df.loc[selected_indices, "log_level"] = np.random.choice(
            ["INFO", "WARN", "ERROR"],
            len(selected_indices),
            p=[0.35, 0.4, 0.25]
        )

    elif anomaly_type == "partial_service_degradation":
        cloud_df.loc[selected_indices, "response_time_ms"] += np.random.normal(
            160, 70, len(selected_indices)
        )
        cloud_df.loc[selected_indices, "cpu_usage"] += np.random.normal(
            10, 6, len(selected_indices)
        )
        cloud_df.loc[selected_indices, "status_code"] = np.random.choice(
            [200, 201, 429, 500],
            len(selected_indices),
            p=[0.55, 0.1, 0.2, 0.15]
        )

cloud_df["response_time_ms"] = cloud_df["response_time_ms"].clip(80, 1200)
cloud_df["cpu_usage"] = cloud_df["cpu_usage"].clip(10, 100)
cloud_df["memory_usage"] = cloud_df["memory_usage"].clip(25, 100)


# 3. Time-window aggregation
cloud_df["timestamp"] = pd.to_datetime(cloud_df["timestamp"])
cloud_df["time_window"] = cloud_df["timestamp"].dt.floor("1min")

window_df = cloud_df.groupby(["time_window", "service"]).agg(
    avg_response_time=("response_time_ms", "mean"),
    max_response_time=("response_time_ms", "max"),
    avg_cpu_usage=("cpu_usage", "mean"),
    max_cpu_usage=("cpu_usage", "max"),
    avg_memory_usage=("memory_usage", "mean"),
    max_memory_usage=("memory_usage", "max"),
    error_rate=("status_code", lambda x: (x >= 500).mean()),
    warn_rate=("log_level", lambda x: (x == "WARN").mean()),
    error_log_rate=("log_level", lambda x: (x == "ERROR").mean()),
    log_count=("label", "count"),
    anomaly_label=("label", "max")
).reset_index()


# 4. Prepare features
window_features = window_df[
    [
        "service",
        "avg_response_time",
        "max_response_time",
        "avg_cpu_usage",
        "max_cpu_usage",
        "avg_memory_usage",
        "max_memory_usage",
        "error_rate",
        "warn_rate",
        "error_log_rate",
        "log_count"
    ]
]

window_target = window_df["anomaly_label"]

window_features_encoded = pd.get_dummies(
    window_features,
    columns=["service"],
    drop_first=True
)


# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    window_features_encoded,
    window_target,
    test_size=0.2,
    random_state=42,
    stratify=window_target
)


# 6. Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    max_depth=8
)

model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)

model_package = {
    "model": model,
    "feature_columns": list(window_features_encoded.columns)
}

joblib.dump(model_package, "models/window_rf_model.pkl")

print("Model saved to models/window_rf_model.pkl")

# 7. Evaluate
y_pred = model.predict(X_test)

print("\nWindow-Based Random Forest Results")
print("----------------------------------")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

false_positive_rate = fp / (fp + tn)

print("Confusion Matrix:")
print(cm)

print(f"\nFalse Positive Rate: {false_positive_rate:.4f}")
print("\nPipeline completed successfully.")