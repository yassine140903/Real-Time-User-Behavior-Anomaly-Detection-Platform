import psycopg2
import os

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "amen_anomaly",
    "user": "postgres",
    "password": "pwd",  # ← your pgAdmin password
}

DATA_DIR = "data"

LOAD_ORDER = [
    ("employees_master", "employees_master.csv"),
    ("clients_master", "clients_master.csv"),
    ("accounts", "accounts.csv"),
    ("transactions", "transactions.csv"),
]

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

for table, filename in LOAD_ORDER:
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r") as f:
        cur.copy_expert(f"COPY {table} FROM STDIN WITH CSV HEADER", f)
    conn.commit()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    count = cur.fetchone()[0]
    print(f"Loaded {count} rows into {table}")

cur.close()
conn.close()
print("Done.")