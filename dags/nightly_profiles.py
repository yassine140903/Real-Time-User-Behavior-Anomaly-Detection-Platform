# dags/nightly_profiles.py

from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

APP_DIR = "/app"

default_args = {
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def validate_profiles(**context):
    import sys
    sys.path.insert(0, "/app")
    import psycopg2
    from src.config import DB_CONFIG

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM profile_snapshots")
        snapshot_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT client_id) FROM profile_snapshots")
        distinct_clients = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM clients_master")
        clients_master_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM profile_snapshots WHERE profile_data IS NULL")
        null_count = cur.fetchone()[0]

        cur.close()
    finally:
        conn.close()

    if snapshot_count <= 0:
        raise AirflowFailException("profile_snapshots is empty — batch_profile produced no rows")

    if clients_master_count > 0:
        coverage = distinct_clients / clients_master_count
        if coverage < 0.95:
            raise AirflowFailException(
                f"Only {distinct_clients}/{clients_master_count} clients "
                f"({coverage:.1%}) have a profile snapshot — expected >= 95%"
            )

    if null_count > 0:
        raise AirflowFailException(f"{null_count} profile_snapshots rows have NULL profile_data")

    print(
        f"validate_profiles OK — {snapshot_count} snapshots, "
        f"{distinct_clients}/{clients_master_count} clients covered, "
        f"0 NULL profile_data rows"
    )


def validate_employee_profiles(**context):
    import sys
    sys.path.insert(0, "/app")
    import psycopg2
    from src.config import DB_CONFIG

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(DISTINCT employee_id) FROM employee_profile_snapshots")
        distinct_employees = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM employees_master")
        employees_master_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM employee_profile_snapshots WHERE profile_data IS NULL")
        null_count = cur.fetchone()[0]

        cur.close()
    finally:
        conn.close()

    if employees_master_count > 0:
        coverage = distinct_employees / employees_master_count
        if coverage < 0.95:
            raise AirflowFailException(
                f"Only {distinct_employees}/{employees_master_count} employees "
                f"({coverage:.1%}) have a profile snapshot — expected >= 95%"
            )

    if null_count > 0:
        raise AirflowFailException(f"{null_count} employee_profile_snapshots rows have NULL profile_data")

    print(
        f"validate_employee_profiles OK — "
        f"{distinct_employees}/{employees_master_count} employees covered, "
        f"0 NULL profile_data rows"
    )


def validate_baselines(**context):
    import sys
    sys.path.insert(0, "/app")
    import psycopg2
    from src.config import DB_CONFIG

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM archetype_baselines")
        baseline_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM archetype_baselines WHERE baseline_data IS NULL")
        null_count = cur.fetchone()[0]

        cur.close()
    finally:
        conn.close()

    if baseline_count != 5:
        raise AirflowFailException(
            f"Expected exactly 5 archetype baselines "
            f"(salaried, small_business, student, retiree, big_business), found {baseline_count}"
        )

    if null_count > 0:
        raise AirflowFailException(f"{null_count} archetype_baselines rows have NULL baseline_data")

    print(f"validate_baselines OK — {baseline_count} baselines, 0 NULL baseline_data rows")


def validate_redis(**context):
    import sys
    sys.path.insert(0, "/app")
    import redis
    from src.config import REDIS_HOST, REDIS_PORT

    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    try:
        profile_client_count = len(list(client.scan_iter(match="profile:client:*")))
        buffer_client_count = len(list(client.scan_iter(match="buffer:client:*")))
        baseline_count = len(list(client.scan_iter(match="baseline:*")))
        profile_employee_count = len(list(client.scan_iter(match="profile:employee:*")))
    finally:
        client.close()

    if profile_client_count <= 0:
        raise AirflowFailException("No profile:client:* keys found in Redis")

    if buffer_client_count <= 0:
        raise AirflowFailException("No buffer:client:* keys found in Redis")

    if baseline_count != 5:
        raise AirflowFailException(f"Expected exactly 5 baseline:* keys in Redis, found {baseline_count}")

    if profile_employee_count <= 0:
        raise AirflowFailException("No profile:employee:* keys found in Redis")

    print(
        f"validate_redis OK — profile:client:*={profile_client_count}, "
        f"buffer:client:*={buffer_client_count}, baseline:*={baseline_count}, "
        f"profile:employee:*={profile_employee_count}"
    )


with DAG(
    dag_id="nightly_profiles",
    schedule="0 2 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["batch", "profiles"],
) as dag:

    run_batch_profile = BashOperator(
        task_id="run_batch_profile",
        bash_command=f"cd {APP_DIR} && python -m src.batch.batch_profile",
    )

    validate_profiles_task = PythonOperator(
        task_id="validate_profiles",
        python_callable=validate_profiles,
    )

    run_batch_employee_profile = BashOperator(
        task_id="run_batch_employee_profile",
        bash_command=f"cd {APP_DIR} && python -m src.batch.batch_employee_profile",
    )

    validate_employee_profiles_task = PythonOperator(
        task_id="validate_employee_profiles",
        python_callable=validate_employee_profiles,
    )

    run_batch_archetype_baselines = BashOperator(
        task_id="run_batch_archetype_baselines",
        bash_command=f"cd {APP_DIR} && python -m src.batch.batch_archetype_baselines",
    )

    validate_baselines_task = PythonOperator(
        task_id="validate_baselines",
        python_callable=validate_baselines,
    )

    run_hydrate_redis = BashOperator(
        task_id="run_hydrate_redis",
        bash_command=f"cd {APP_DIR} && python -m src.batch.hydrate_redis",
    )

    validate_redis_task = PythonOperator(
        task_id="validate_redis",
        python_callable=validate_redis,
    )

    (
        run_batch_profile
        >> validate_profiles_task
        >> run_batch_employee_profile
        >> validate_employee_profiles_task
        >> run_batch_archetype_baselines
        >> validate_baselines_task
        >> run_hydrate_redis
        >> validate_redis_task
    )
