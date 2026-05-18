# Cloud Anomaly Detection System

A Dockerised machine learning project for detecting anomalies in simulated cloud telemetry using a window-based Random Forest model and FastAPI.

## Features

- Simulated cloud telemetry generation
- Window-based behavioural aggregation
- Random Forest anomaly detection model
- Probabilistic anomaly scoring
- Configurable anomaly threshold
- Dockerised FastAPI inference API
- Live telemetry simulator

## Tech Stack

- Python
- Scikit-learn
- Pandas
- FastAPI
- Docker
- Random Forest

## Run with Docker

Build the image:

```bash
docker build -t cloud-anomaly-detector-api .