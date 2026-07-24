# dags/feedback_monitor.py
"""
Daily check: if supervisor rejection rate on ALERT+BLOCK exceeds 50%
over the past 7 days (min 10 reviews), trigger weekly_retrain DAG.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.api.common.trigger_dag import trigger_dag
from datetime import datetime, timedelta
import psycopg2
import os

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

REJECTION_THRESHOLD = 0.50
MIN_REVIEWS = 10

def check_feedback_divergence(**context):
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "amen_anomaly"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE supervisor_decision = 'rejected') AS rejected,
                COUNT(*) FILTER (WHERE supervisor_decision IS NOT NULL) AS total_reviewed
            FROM alerts
            WHERE alert_tier IN ('ALERT', 'BLOCK')
              AND decision_timestamp >= NOW() - INTERVAL '7 days'
        """)
        rejected, total_reviewed = cur.fetchone()
    
    conn.close()

    if total_reviewed < MIN_REVIEWS:
        print(f"Not enough reviews: {total_reviewed} < {MIN_REVIEWS}. Skipping.")
        return

    rejection_rate = rejected / total_reviewed
    print(f"Rejection rate: {rejected}/{total_reviewed} = {rejection_rate:.2%}")

    if rejection_rate > REJECTION_THRESHOLD:
        print(f"Threshold exceeded ({rejection_rate:.2%} > {REJECTION_THRESHOLD:.0%}). Triggering retrain.")
        trigger_dag(
            dag_id="weekly_retrain",
            conf={"trigger": "feedback_divergence"},
            run_id=f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    else:
        print(f"Within acceptable range. No retrain needed.")

with DAG(
    dag_id="feedback_monitor",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["monitoring", "feedback"],
) as dag:

    check_feedback = PythonOperator(
        task_id="check_feedback_divergence",
        python_callable=check_feedback_divergence,
    )