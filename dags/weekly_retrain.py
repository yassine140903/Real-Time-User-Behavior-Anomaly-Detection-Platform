# dags/weekly_retrain.py

from datetime import datetime, timedelta

from docker.types import Mount

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator

APP_DIR = "/app"

# Project root on the HOST machine (not inside the airflow container).
# DockerOperator talks to the host Docker daemon via the mounted socket, so bind-mount
# sources must be host paths, not the /app path this container sees them at.
# Set this to the actual project directory on the host running docker-compose.
PROJECT_DIR = r"C:\Users\benje\Iter\Real-Time-User-Behavior-Anomaly-Detection-Platform"

# Docker Compose network name so the training container can reach `postgres`.
# Compose names it "{project_folder}_default" by default (lowercased). For this
# repo's folder that's "real-time-user-behavior-anomaly-detection-platform_default" —
# update if COMPOSE_PROJECT_NAME is overridden on the host.
COMPOSE_NETWORK = "real-time-user-behavior-anomaly-detection-platform_default"

default_args = {
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def validate_enrichment(**context):
    import sys
    sys.path.insert(0, "/app")
    import json
    import pandas as pd

    df = pd.read_csv("/app/data/training_features.csv")

    with open("/app/models/feature_cols.json") as f:
        feature_cols = json.load(f)

    if len(df) <= 0:
        raise AirflowFailException("data/training_features.csv is empty")

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise AirflowFailException(f"training_features.csv is missing expected feature columns: {missing_cols}")

    nan_counts = df[feature_cols].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"WARNING: NaN values found in {len(nan_cols)} feature columns:\n{nan_cols}")

        nan_rates = nan_cols / len(df)
        over_threshold = nan_rates[nan_rates > 0.05]
        if len(over_threshold) > 0:
            raise AirflowFailException(
                f"NaN rate exceeds 5% in columns: {over_threshold.to_dict()}"
            )

    print(f"validate_enrichment OK — shape={df.shape}, {len(feature_cols)} feature columns present")


with DAG(
    dag_id="weekly_retrain",
    schedule="0 4 * * 0",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["batch", "retrain"],
) as dag:

    run_batch_enrichment = BashOperator(
        task_id="run_batch_enrichment",
        bash_command=f"cd {APP_DIR} && python -m src.batch.batch_enrichment",
    )

    validate_enrichment_task = PythonOperator(
        task_id="validate_enrichment",
        python_callable=validate_enrichment,
    )

    retrain_models = DockerOperator(
        task_id="retrain_models",
        image="amen-anomaly-training:latest",
        force_pull=False,
        command="python -m src.training.train_all",
        docker_url="unix://var/run/docker.sock",
        network_mode=COMPOSE_NETWORK,
        auto_remove=True,
        mounts=[
            Mount(source=f"{PROJECT_DIR}/data", target="/app/data", type="bind"),
            Mount(source=f"{PROJECT_DIR}/models", target="/app/models", type="bind"),
        ],
        mount_tmp_dir=False,
        environment={
            "DB_HOST": "postgres",
            "DB_PORT": "5432",
            "DB_NAME": "amen_anomaly",
            "DB_USER": "postgres",
            "DB_PASSWORD": "postgres",
        },
    )

    run_batch_enrichment >> validate_enrichment_task >> retrain_models
