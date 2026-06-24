"""
One-time script: load generator CSVs into PostgreSQL (clean slate).
Assumes tables already created via the DDL script.
"""

import pandas as pd
import psycopg2
from io import StringIO

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "amen_anomaly",
    "user": "postgres",
    "password": "9011361923257228",
}

DATA_DIR = "data"


def copy_df(conn, df, table_name):
    """Fast bulk load via PostgreSQL COPY protocol."""
    buf = StringIO()
    df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    
    cols = ", ".join(df.columns)
    cur = conn.cursor()
    cur.copy_expert(f"COPY {table_name} ({cols}) FROM STDIN WITH CSV", buf)
    conn.commit()
    cur.close()
    print(f"  {table_name}: {len(df)} rows loaded")


def main():
    # ---- Read CSVs ----
    print("Reading CSVs...")
    clients_df = pd.read_csv(f"{DATA_DIR}/clients.csv")
    tx_df = pd.read_csv(f"{DATA_DIR}/transactions.csv")
    
    # ---- 1. clients_master ----
    print("Loading clients_master...")
    cm = clients_df[["client_id", "archetype", "account_opening_date", "home_branch"]].copy()
    cm = cm.rename(columns={"home_branch": "home_branch_id"})
    cm["nationality"] = "TN"
    cm["client_type"] = cm["archetype"].map(
        lambda a: "personne_morale" if a in ("small_business", "big_business") else "personne_physique"
    )
    # Reorder to match table columns
    cm = cm[["client_id", "archetype", "account_opening_date", "nationality", "client_type", "home_branch_id"]]
    
    conn = psycopg2.connect(**DB_CONFIG)
    copy_df(conn, cm, "clients_master")
    
    # ---- 2. accounts ----
    print("Loading accounts...")
    acc = clients_df[["account_id", "client_id", "account_opening_date"]].copy()
    acc = acc.rename(columns={"account_opening_date": "opening_date"})
    acc["account_type"] = "courant"
    acc["status"] = "active"
    acc = acc[["account_id", "client_id", "account_type", "opening_date", "status"]]
    
    copy_df(conn, acc, "accounts")
    
    # ---- 3. employees_master ----
    print("Loading employees_master...")
    emp = tx_df[["employee_id", "branch_id"]].drop_duplicates()
    
    copy_df(conn, emp, "employees_master")
    
    # ---- 4. transactions ----
    print("Loading transactions...")
    tx_cols = [
        "event_id", "client_id", "account_id", "employee_id",
        "branch_id", "timestamp", "amount", "currency", "channel",
        "operation_type", "payload", "is_anomaly", "anomaly_type"
    ]
    tx = tx_df[tx_cols].copy()
    
    copy_df(conn, tx, "transactions")
    
    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()