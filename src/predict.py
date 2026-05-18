import joblib
import pandas as pd


model_package = joblib.load("models/window_rf_model.pkl")

model = model_package["model"]
feature_columns = model_package["feature_columns"]


sample = {
    "service": "payment-service",
    "avg_response_time": 950,
    "max_response_time": 1400,
    "avg_cpu_usage": 90,
    "max_cpu_usage": 99,
    "avg_memory_usage": 92,
    "max_memory_usage": 98,
    "error_rate": 0.45,
    "warn_rate": 0.55,
    "error_log_rate": 0.35,
    "log_count": 45
}

input_df = pd.DataFrame([sample])

input_encoded = pd.get_dummies(
    input_df,
    columns=["service"],
    drop_first=True
)

input_encoded = input_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)

probability = model.predict_proba(input_encoded)[0]

anomaly_probability = probability[1]

threshold = 0.40

if anomaly_probability >= threshold:
    print("Prediction: ANOMALOUS")
else:
    print("Prediction: NORMAL")

print("\nClass probabilities:")
print(f"Normal: {probability[0]:.4f}")
print(f"Anomaly: {probability[1]:.4f}")

print(f"\nThreshold used: {threshold}")