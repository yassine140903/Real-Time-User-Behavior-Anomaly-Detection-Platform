# src/batch/batch_enrichment.py

import csv
import json
import psycopg2
import psycopg2.extras
import pandas as pd

from src.config import DB_CONFIG
from src.enrichment.core import EnrichmentCore, BUFFER_SIZE, EMP_BUFFER_SIZE

OUTPUT_PATH = "data/training_features.csv"
CHUNK_SIZE = 5000
LOG_EVERY = 10_000


def load_profile_history(conn):
    """client_id -> list of (computed_at, profile_dict) sorted ascending by time."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT client_id, computed_at, profile_data FROM profile_snapshots "
        "ORDER BY client_id, computed_at"
    )

    history = {}
    for row in cur:
        cid = str(row["client_id"])
        data = row["profile_data"]
        data = data if isinstance(data, dict) else json.loads(data)
        data.pop("recent_events_buffer", None)
        history.setdefault(cid, []).append((row["computed_at"], data))

    cur.close()
    print(f"Loaded profile snapshot history for {len(history)} clients")
    return history


def load_archetype_baselines(conn):
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT archetype, baseline_data FROM archetype_baselines")
    baselines = {}
    for row in cur:
        data = row["baseline_data"]
        baselines[row["archetype"]] = data if isinstance(data, dict) else json.loads(data)
    cur.close()
    print(f"Loaded {len(baselines)} archetype baselines")
    return baselines


def load_labels(conn):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT event_id, is_anomaly, anomaly_type FROM evaluation_labels")
        labels = {
            str(row["event_id"]): (row["is_anomaly"], row["anomaly_type"])
            for row in cur
        }
        cur.close()
        print(f"Loaded {len(labels)} evaluation labels")
        return labels
    except Exception:
        print("No evaluation_labels table found — continuing without labels")
        return {}


def find_profile(history_entries, ts):
    """Latest snapshot strictly before ts, or None."""
    if not history_entries:
        return None
    prof = None
    for computed_at, data in history_entries:
        if computed_at < ts:
            prof = data
        else:
            break
    return prof


def run_enrichment():
    conn = psycopg2.connect(**DB_CONFIG)

    print("Loading reference data...")
    profile_history = load_profile_history(conn)
    baselines = load_archetype_baselines(conn)
    labels = load_labels(conn)

    core = EnrichmentCore(redis_client=None, models_dir="models")

    # Prepare CSV writer — write header first, then append rows in chunks
    feature_cols = core.feature_cols
    output_cols = feature_cols + ["event_id", "client_id", "is_anomaly", "anomaly_type","timestamp"]

    csv_file = open(OUTPUT_PATH, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=output_cols, extrasaction="ignore")
    writer.writeheader()

    # Server-side cursor — streams rows without loading all into memory
    tx_cursor = conn.cursor(name="tx_stream", cursor_factory=psycopg2.extras.RealDictCursor)
    tx_cursor.execute(
        "SELECT event_id, client_id, account_id, employee_id, branch_id, "
        "timestamp, amount, currency, channel, operation_type, payload "
        "FROM transactions ORDER BY timestamp"
    )

    client_buffers = {}
    employee_buffers = {}
    processed = 0

    while True:
        rows = tx_cursor.fetchmany(CHUNK_SIZE)
        if not rows:
            break

        chunk_results = []

        for row in rows:
            cid = str(row["client_id"])
            eid = row["employee_id"]
            ts = pd.Timestamp(row["timestamp"])
            amount = float(row["amount"])
            op = row["operation_type"]

            event = {
                "event_id": str(row["event_id"]),
                "client_id": cid,
                "account_id": str(row["account_id"]),
                "employee_id": eid,
                "branch_id": row["branch_id"],
                "timestamp": ts,
                "amount": amount,
                "operation_type": op,
                "payload": row["payload"],
            }

            prof = find_profile(profile_history.get(cid, []), ts)
            arch = prof.get("archetype", "salaried") if prof else "salaried"
            archetype_baseline = baselines.get(arch)

            cbuf = client_buffers.setdefault(cid, [])
            ebuf = employee_buffers.setdefault(eid, [])

            features = core._compute_features(
                event, prof, cbuf, ebuf, ts, amount, op, eid,
                archetype_baseline=archetype_baseline
            )

            is_anomaly, anomaly_type = labels.get(str(row["event_id"]), (None, None))
            features["event_id"] = str(row["event_id"])
            features["client_id"] = cid
            features["is_anomaly"] = is_anomaly
            features["anomaly_type"] = anomaly_type
            features["timestamp"] = ts
            chunk_results.append(features)

            # Update buffers
            cbuf.append({"ts": ts, "amount": amount, "op": op, "eid": eid, "payload": row["payload"]})
            if len(cbuf) > BUFFER_SIZE:
                del cbuf[:-BUFFER_SIZE]

            ebuf.append({"ts": ts, "cid": cid})
            if len(ebuf) > EMP_BUFFER_SIZE:
                del ebuf[:-EMP_BUFFER_SIZE]

        # Write chunk to CSV
        writer.writerows(chunk_results)
        processed += len(chunk_results)

        if processed % LOG_EVERY < CHUNK_SIZE:
            print(f"Processed {processed} events")

    tx_cursor.close()
    csv_file.close()
    conn.close()

    print(f"Done — {processed} rows written to {OUTPUT_PATH}")

    # Quick summary
    df = pd.read_csv(OUTPUT_PATH, nrows=5)
    print(f"Output columns: {len(df.columns)}")
    print(f"Sample columns: {list(df.columns[:5])}...")


if __name__ == "__main__":
    run_enrichment()
