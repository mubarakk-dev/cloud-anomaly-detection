import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# Load trained model package
model_package = joblib.load("models/window_rf_model.pkl")

model = model_package["model"]
feature_columns = model_package["feature_columns"]

app = FastAPI(
    title="Cloud Anomaly Detection API",
    description="A Dockerised machine learning API for cloud log anomaly detection.",
    version="1.0"
)


class ServiceWindow(BaseModel):
    service: str
    avg_response_time: float
    max_response_time: float
    avg_cpu_usage: float
    max_cpu_usage: float
    avg_memory_usage: float
    max_memory_usage: float
    error_rate: float
    warn_rate: float
    error_log_rate: float
    log_count: int


@app.get("/")
def home():
    return {
        "message": "Cloud Anomaly Detection API is running."
    }


@app.post("/predict")
def predict(data: ServiceWindow):
    input_df = pd.DataFrame([data.dict()])

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
        prediction = "ANOMALOUS"
    else:
        prediction = "NORMAL"

    return {
        "prediction": prediction,
        "normal_probability": round(float(probability[0]), 4),
        "anomaly_probability": round(float(anomaly_probability), 4),
        "threshold": threshold
    }