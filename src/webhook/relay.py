import os
import requests
from fastapi import FastAPI, Request

app = FastAPI()

AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://airflow-webserver:8080")

@app.post("/trigger")
async def trigger_retrain(request: Request):
    """Receives AlertManager webhook, triggers Airflow DAG."""
    response = requests.post(
        f"{AIRFLOW_URL}/api/v1/dags/weekly_retrain/dagRuns",
        json={"conf": {"trigger": "drift_alert"}},
        auth=("admin", os.getenv("AIRFLOW_ADMIN_PASSWORD", "admin")),
        headers={"Content-Type": "application/json"},
    )
    return {"status": response.status_code, "airflow_response": response.json()}