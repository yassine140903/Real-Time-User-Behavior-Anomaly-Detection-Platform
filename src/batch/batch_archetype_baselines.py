import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
import json
from src.config import DB_CONFIG

OPERATIONS = ["retrait", "versement", "virement", "cheque"]


def compute_archetype_baselines(conn):
    # Join transactions with clients_master to get archetype per transaction
    query = """
        SELECT t.*, c.archetype
        FROM transactions t
        JOIN clients_master c ON t.client_id = c.client_id
    """
    df = pd.read_sql(query, conn, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} transactions with archetype labels")

    baselines = {}

    for archetype, arch_txs in df.groupby("archetype"):
        baseline = {}

        # Amount stats per operation
        for op in OPERATIONS:
            op_txs = arch_txs[arch_txs["operation_type"] == op]
            prefix = f"amount_{op}"

            if len(op_txs) == 0:
                baseline[f"{prefix}_mean"] = 0.0
                baseline[f"{prefix}_std"] = 0.0
            else:
                baseline[f"{prefix}_mean"] = round(float(op_txs["amount"].mean()), 2)
                baseline[f"{prefix}_std"] = round(float(op_txs["amount"].std() if len(op_txs) > 1 else 0.0), 2)

        # Hour distribution
        hours = arch_txs["timestamp"].dt.hour
        hour_dist = hours.value_counts(normalize=True).sort_index()
        baseline["hour_distribution"] = {str(k): round(v, 4) for k, v in hour_dist.items()}

        # Day-of-week distribution
        dow = arch_txs["timestamp"].dt.dayofweek
        dow_dist = dow.value_counts(normalize=True).sort_index()
        baseline["dow_distribution"] = {str(k): round(v, 4) for k, v in dow_dist.items()}

        # Branch distribution
        branch_dist = arch_txs["branch_id"].value_counts(normalize=True)
        baseline["branch_distribution"] = {k: round(v, 4) for k, v in branch_dist.items()}

        # Operation mix
        op_dist = arch_txs["operation_type"].value_counts(normalize=True)
        baseline["operation_distribution"] = {k: round(v, 4) for k, v in op_dist.items()}

        baselines[archetype] = baseline
        print(f"  {archetype}: {len(arch_txs)} transactions")

    # Store in archetype_baselines table
    cur = conn.cursor()
    for archetype, baseline in baselines.items():
        cur.execute(
            "INSERT INTO archetype_baselines (archetype, baseline_data) VALUES (%s, %s::jsonb) "
            "ON CONFLICT (archetype) DO UPDATE SET baseline_data = EXCLUDED.baseline_data",
            (archetype, json.dumps(baseline))
        )
    conn.commit()
    cur.close()
    print(f"Done — {len(baselines)} archetype baselines written.")


if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)
    compute_archetype_baselines(conn)
    conn.close()