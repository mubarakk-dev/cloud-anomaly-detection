# Cloud Anomaly Detection System

A Dockerised machine learning-based anomaly detection system for simulated cloud telemetry using behavioural aggregation, FastAPI, and Random Forest classification.

---

# Overview

Modern cloud environments generate large volumes of operational telemetry, making manual monitoring increasingly difficult. This project demonstrates a practical machine learning pipeline for detecting anomalous cloud-service behaviour using simulated real-time operational metrics.

The system combines:

* behavioural time-window aggregation
* machine learning-based anomaly detection
* probabilistic anomaly scoring
* Docker containerisation
* FastAPI inference services
* simulated live cloud telemetry streaming

The project was originally developed as part of an MSc research project focused on anomaly detection within distributed cloud environments.

---

# Features

## Machine Learning

* Window-based behavioural anomaly detection
* Random Forest classification model
* Probabilistic anomaly scoring
* Configurable anomaly thresholds
* Realistic cloud-style telemetry simulation

## Cloud / DevOps / Deployment

* Dockerised API deployment
* FastAPI REST inference service
* Live telemetry simulator
* Reproducible containerised environment
* Real-time anomaly prediction workflow

## Operational Telemetry

The system models realistic cloud operational behaviour including:

* response latency
* CPU utilisation
* memory utilisation
* service-level behaviour
* warning/error frequencies
* operational degradation patterns

---

# System Architecture

```text
Live Cloud Telemetry Simulation
                ↓
Behavioural Time-Window Aggregation
                ↓
Random Forest Anomaly Detection Model
                ↓
FastAPI Inference Service
                ↓
Probabilistic Risk Scoring
                ↓
Real-Time Anomaly Alerts
```

---

# Project Structure

```text
cloud-anomaly-detection/
│
├── src/
│   ├── api.py
│   ├── predict.py
│   ├── live_simulator.py
│   └── window_based_detection.py
│
├── models/
│   └── window_rf_model.pkl
│
├── data/
│
├── outputs/
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* FastAPI
* Docker
* Uvicorn
* Joblib

---

# Machine Learning Pipeline

The anomaly detection pipeline follows these stages:

1. Simulated cloud telemetry generation
2. Behavioural time-window aggregation
3. Operational feature engineering
4. Random Forest model training
5. Probabilistic anomaly scoring
6. Threshold-based anomaly classification
7. Real-time inference through FastAPI

---

# Operational Features

The final model uses behavioural operational features including:

* average response time
* maximum response time
* average CPU usage
* maximum CPU usage
* average memory usage
* maximum memory usage
* error-rate frequency
* warning-rate frequency
* error log ratio
* operational log volume
* service-level context

---

# Example API Input

```json
{
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
```

---

# Example API Output

```json
{
  "prediction": "ANOMALOUS",
  "normal_probability": 0.4793,
  "anomaly_probability": 0.5207,
  "threshold": 0.4
}
```

---

# Docker Setup

## Build the Docker Image

```bash
docker build -t cloud-anomaly-detector-api .
```

---

## Run the API Container

```bash
docker run -p 8000:8000 -v ${PWD}/models:/app/models cloud-anomaly-detector-api
```

---

# FastAPI Documentation

Once running:

```text
http://localhost:8000/docs
```

The Swagger UI allows interactive testing of anomaly predictions.

---

# Live Telemetry Simulation

Run the simulator locally:

```bash
python src/live_simulator.py
```

This continuously sends simulated cloud-service telemetry to the Dockerised API for real-time anomaly scoring.

---

# Example Real-Time Output

```text
Service: database-service
Expected pattern: ANOMALY-LIKE
API prediction: ANOMALOUS
Anomaly probability: 0.5207
Threshold: 0.4
```

---

# Key Learning Outcomes

This project demonstrates practical experience with:

* machine learning engineering
* cloud-style telemetry analysis
* behavioural anomaly detection
* Docker containerisation
* FastAPI deployment
* real-time inference systems
* probabilistic operational monitoring
* ML deployment workflows

---

# Future Improvements

Potential future extensions include:

* Kafka-based streaming pipelines
* Grafana monitoring dashboards
* Prometheus integration
* Kubernetes deployment
* online learning models
* cloud deployment on AWS/Azure/GCP
* distributed anomaly correlation

---

# Author

Khaled Mubarak

GitHub: [https://github.com/mubarakk-dev](https://github.com/mubarakk-dev)
