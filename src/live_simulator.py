import time
import random
import requests


API_URL = "http://localhost:8000/predict"


services = [
    "api-gateway",
    "auth-service",
    "payment-service",
    "inventory-service",
    "database-service",
    "cache-service"
]


def generate_normal_window():
    return {
        "service": random.choice(services),
        "avg_response_time": random.randint(180, 380),
        "max_response_time": random.randint(300, 600),
        "avg_cpu_usage": random.randint(35, 65),
        "max_cpu_usage": random.randint(55, 78),
        "avg_memory_usage": random.randint(50, 75),
        "max_memory_usage": random.randint(65, 85),
        "error_rate": round(random.uniform(0.00, 0.08), 2),
        "warn_rate": round(random.uniform(0.02, 0.12), 2),
        "error_log_rate": round(random.uniform(0.00, 0.05), 2),
        "log_count": random.randint(20, 45)
    }


def generate_anomalous_window():
    return {
        "service": random.choice(["payment-service", "database-service", "api-gateway"]),
        "avg_response_time": random.randint(650, 1100),
        "max_response_time": random.randint(900, 1600),
        "avg_cpu_usage": random.randint(70, 95),
        "max_cpu_usage": random.randint(85, 100),
        "avg_memory_usage": random.randint(75, 95),
        "max_memory_usage": random.randint(85, 100),
        "error_rate": round(random.uniform(0.18, 0.50), 2),
        "warn_rate": round(random.uniform(0.25, 0.60), 2),
        "error_log_rate": round(random.uniform(0.10, 0.40), 2),
        "log_count": random.randint(30, 70)
    }


print("Starting live cloud log simulator...")
print("Sending service-window telemetry to API every 3 seconds.")
print("Press CTRL+C to stop.\n")


while True:
    try:
        # 80% normal, 20% anomalous
        if random.random() < 0.8:
            payload = generate_normal_window()
            expected = "NORMAL-LIKE"
        else:
            payload = generate_anomalous_window()
            expected = "ANOMALY-LIKE"

        response = requests.post(API_URL, json=payload)
        result = response.json()

        print("=" * 70)
        print(f"Service: {payload['service']}")
        print(f"Expected pattern: {expected}")
        print(f"API prediction: {result['prediction']}")
        print(f"Anomaly probability: {result['anomaly_probability']}")
        print(f"Threshold: {result['threshold']}")

        time.sleep(3)

    except KeyboardInterrupt:
        print("\nSimulator stopped.")
        break

    except Exception as e:
        print("Error sending request:", e)
        time.sleep(3)