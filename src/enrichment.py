"""
Contrast-focused enrichment pipeline.

Produces ~30 features per event:
  - Raw event features (amount, hour, op_type one-hot, branch match)
  - Contrast features (event vs client baseline: z-scores, ratios, flags)
  - Velocity features (from rolling buffer: tx counts, cumulative amounts, duplicates)
  - Employee features (from buffer: employee tx count, employee-client pair count)
  - Minimal context (archetype one-hot, account age)
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict


OPS = ["retrait", "versement", "virement", "cheque"]
ARCHETYPES = ["salaried", "small_business", "student", "retiree", "big_business"]
BUFFER_SIZE = 50
EMP_BUFFER_SIZE = 100


def enrich(transactions_path: str, clients_path: str, output_path: str,
           start_date: str = "2025-01-01"):
    """Run the full enrichment pipeline."""

    print("Loading data...")
    tx = pd.read_csv(transactions_path)
    tx = tx.sort_values("timestamp").reset_index(drop=True)

    clients = pd.read_csv(clients_path).set_index("client_id")

    sim_start = pd.Timestamp(start_date)

    # Rolling buffers — lists of lightweight dicts
    client_buf = defaultdict(list)   # client_id -> [{ts, amount, op, eid, payload}]
    emp_buf = defaultdict(list)      # employee_id -> [{ts, cid}]

    enriched = []
    n = len(tx)

    for i, row in tx.iterrows():
        if i % 50000 == 0:
            print(f"  Enriching {i}/{n} ...")

        cid = row["client_id"]
        eid = row["employee_id"]
        ts = pd.Timestamp(row["timestamp"])
        amount = float(row["amount"])
        op = row["operation_type"]

        # Client baseline (oracle profile)
        prof = clients.loc[cid] if cid in clients.index else None

        # Current buffers (BEFORE adding this event)
        cbuf = client_buf[cid]
        ebuf = emp_buf[eid]

        f = _compute_features(row, prof, cbuf, ebuf, ts, amount, op, eid, sim_start)

        # Preserve identifiers + labels
        f["event_id"] = row["event_id"]
        f["client_id"] = cid
        f["timestamp"] = row["timestamp"]
        f["is_anomaly"] = row["is_anomaly"]
        f["anomaly_type"] = row["anomaly_type"] if pd.notna(row.get("anomaly_type")) else None
        f["difficulty"] = row["difficulty"] if pd.notna(row.get("difficulty")) else None

        enriched.append(f)

        # Update buffers AFTER feature computation
        cbuf.append({"ts": ts, "amount": amount, "op": op, "eid": eid,
                      "payload": row["payload"]})
        if len(cbuf) > BUFFER_SIZE:
            client_buf[cid] = cbuf[-BUFFER_SIZE:]

        ebuf.append({"ts": ts, "cid": cid})
        if len(ebuf) > EMP_BUFFER_SIZE:
            emp_buf[eid] = ebuf[-EMP_BUFFER_SIZE:]

    df = pd.DataFrame(enriched)
    df.to_csv(output_path, index=False)
    print(f"Enriched {len(df)} events -> {output_path}")

    # Quick feature summary
    feature_cols = [c for c in df.columns if c not in
                    ("event_id", "client_id", "timestamp", "is_anomaly",
                     "anomaly_type", "difficulty")]
    print(f"Feature count: {len(feature_cols)}")
    print(f"Anomaly events: {df['is_anomaly'].sum()}")
    print(f"Anomaly rate: {df['is_anomaly'].mean()*100:.3f}%")

    return df


def _compute_features(row, prof, cbuf, ebuf, ts, amount, op, eid, sim_start):
    f = {}

    # ── RAW EVENT ──────────────────────────────────────────────────────
    f["amount"] = amount
    f["hour"] = ts.hour

    for o in OPS:
        f[f"op_{o}"] = 1 if op == o else 0

    if prof is not None:
        f["is_home_branch"] = 1 if row["branch_id"] == prof["home_branch"] else 0
    else:
        f["is_home_branch"] = 1

    # ── AMOUNT CONTRAST ────────────────────────────────────────────────
    if prof is not None:
        p_mean = prof.get(f"amount_mean_{op}", amount)
        p_std = prof.get(f"amount_std_{op}", 1.0)
        p_mean = float(p_mean) if not pd.isna(p_mean) else amount
        p_std = float(p_std) if not pd.isna(p_std) else 1.0
        f["z_amount"] = (amount - p_mean) / max(p_std, 1.0)
        f["amount_to_mean_ratio"] = amount / max(p_mean, 1.0)
    else:
        f["z_amount"] = 0.0
        f["amount_to_mean_ratio"] = 1.0

    f["is_above_threshold"] = 1 if amount > 10000 else 0
    f["is_near_threshold"] = 1 if 8000 <= amount <= 9999 else 0
    f["is_round_amount"] = 1 if amount >= 1000 and abs(amount % 1000) < 1 else 0
    f["is_near_cheque_ceiling"] = 1 if 25000 <= amount <= 30000 else 0

    # ── OPERATION CONTRAST ─────────────────────────────────────────────
    if prof is not None:
        f["op_type_probability"] = float(prof.get(f"op_mix_{op}", 0.25))
    else:
        f["op_type_probability"] = 0.25

    # ── TIMING CONTRAST ────────────────────────────────────────────────
    f["is_outside_hours"] = 1 if ts.hour < 8 or ts.hour >= 16 else 0

    if prof is not None:
        pday = int(prof.get("preferred_day", 15))
        f["day_distance_from_preferred"] = abs(ts.day - pday)
    else:
        f["day_distance_from_preferred"] = 0

    # ── COUNTERPARTY CONTRAST ──────────────────────────────────────────
    try:
        payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
    except (json.JSONDecodeError, TypeError):
        payload = {}

    if op == "virement":
        ben = payload.get("beneficiary_id", "")
        f["is_new_counterparty"] = 1 if ben and str(ben).startswith("NEW-") else 0
    elif op == "cheque":
        emt = payload.get("emitter_id", "")
        f["is_new_counterparty"] = 1 if emt and str(emt).startswith("NEW-") else 0
    else:
        f["is_new_counterparty"] = 0

    # ── VELOCITY FROM CLIENT BUFFER ────────────────────────────────────
    t_24h = ts - pd.Timedelta(hours=24)
    t_7d = ts - pd.Timedelta(days=7)

    recent_24h = [e for e in cbuf if e["ts"] >= t_24h]
    recent_7d = [e for e in cbuf if e["ts"] >= t_7d]

    f["tx_count_24h"] = len(recent_24h)
    f["tx_count_7d"] = len(recent_7d)
    f["cumulative_amount_24h"] = sum(e["amount"] for e in recent_24h)

    # Duplicate: same op + similar amount (within 10%) within 24h
    has_dup = 0
    for e in recent_24h:
        if e["op"] == op and abs(e["amount"] - amount) / max(amount, 1) < 0.10:
            has_dup = 1
            break
    f["has_duplicate_recent"] = has_dup

    # Near-threshold deposits in last 7 days (smurfing signal)
    f["near_threshold_count_7d"] = sum(
        1 for e in recent_7d if 8000 <= e["amount"] <= 9999
    )

    # ── EMPLOYEE FROM BUFFER ───────────────────────────────────────────
    emp_24h = [e for e in ebuf if e["ts"] >= t_24h]
    f["employee_tx_count_24h"] = len(emp_24h)

    # Same employee-client pair in 24h
    f["same_employee_client_count_24h"] = sum(
        1 for e in cbuf if e["ts"] >= t_24h and e.get("eid") == eid
    )

    # ── CONTEXT ────────────────────────────────────────────────────────
    if prof is not None:
        arch = prof.get("archetype", "salaried")
        for a in ARCHETYPES:
            f[f"arch_{a}"] = 1 if arch == a else 0
        f["account_age_days"] = (ts - sim_start).days
    else:
        for a in ARCHETYPES:
            f[f"arch_{a}"] = 0
        f["account_age_days"] = 0

    return f


if __name__ == "__main__":
    enrich(
        transactions_path="data/transactions.csv",
        clients_path="data/clients.csv",
        output_path="data/enriched_events.csv",
    )