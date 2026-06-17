import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "amen_anomaly",
    "user": "postgres",
    "password": "pwd",
}

SIM_START = datetime(2025, 1, 1)
SIM_END = datetime(2025, 6, 29)

WINDOWS = {
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "90d": timedelta(days=90),
}

OPERATIONS = ["retrait", "versement", "virement", "cheque"]

NEAR_THRESHOLD_BAND = (8000, 9999)


def generate_weekly_dates(start, end):
    dates = []
    current = start + timedelta(days=7)
    while current <= end:
        dates.append(current)
        current += timedelta(days=7)
    if dates[-1] != end:
        dates.append(end)
    return dates


def fetch_all_transactions(conn):
    query = "SELECT * FROM transactions"
    df = pd.read_sql(query, conn, parse_dates=["timestamp"])
    return df

def compute_transaction_stats(emp_txs, ref_date):
    """
    Same logic as client profile — 4 ops × 4 windows × 6 stats = 96 fields.
    Only difference: these are transactions PROCESSED by the employee,
    not transactions belonging to a client.
    """
    stats = {}
    
    for window_name, window_delta in WINDOWS.items():
        cutoff = ref_date - window_delta
        windowed = emp_txs[emp_txs["timestamp"] >= cutoff]
        
        for op in OPERATIONS:
            op_txs = windowed[windowed["operation_type"] == op]
            prefix = f"tx_{op}_{window_name}"
            
            if len(op_txs) == 0:
                stats[f"{prefix}_count"] = 0
                stats[f"{prefix}_sum"] = 0.0
                stats[f"{prefix}_mean"] = 0.0
                stats[f"{prefix}_std"] = 0.0
                stats[f"{prefix}_min"] = 0.0
                stats[f"{prefix}_max"] = 0.0
            else:
                amounts = op_txs["amount"]
                stats[f"{prefix}_count"] = int(len(op_txs))
                stats[f"{prefix}_sum"] = round(float(amounts.sum()), 2)
                stats[f"{prefix}_mean"] = round(float(amounts.mean()), 2)
                stats[f"{prefix}_std"] = round(float(amounts.std() if len(op_txs) > 1 else 0.0), 2)
                stats[f"{prefix}_min"] = round(float(amounts.min()), 2)
                stats[f"{prefix}_max"] = round(float(amounts.max()), 2)
    
    return stats

def compute_behavioral_patterns(emp_txs, ref_date):
    """
    Employee behavioral patterns. Same 5 distributions as client,
    plus client_distribution (which clients does this employee serve).
    """
    patterns = {}
    
    for window_name, days in [("30d", 30), ("180d", 180)]:
        cutoff = ref_date - timedelta(days=days)
        windowed = emp_txs[emp_txs["timestamp"] >= cutoff]
        
        prefix = f"dist_{window_name}"
        
        if len(windowed) == 0:
            patterns[f"{prefix}_operation"] = {}
            patterns[f"{prefix}_hour"] = {}
            patterns[f"{prefix}_day_of_week"] = {}
            patterns[f"{prefix}_branch"] = {}
            patterns[f"{prefix}_amount_buckets"] = {}
            patterns[f"{prefix}_client"] = {}
            continue
        
        op_counts = windowed["operation_type"].value_counts(normalize=True)
        patterns[f"{prefix}_operation"] = {k: round(v, 4) for k, v in op_counts.items()}
        
        hours = windowed["timestamp"].dt.hour
        hour_counts = hours.value_counts(normalize=True).sort_index()
        patterns[f"{prefix}_hour"] = {str(k): round(v, 4) for k, v in hour_counts.items()}
        
        dow = windowed["timestamp"].dt.dayofweek
        dow_counts = dow.value_counts(normalize=True).sort_index()
        patterns[f"{prefix}_day_of_week"] = {str(k): round(v, 4) for k, v in dow_counts.items()}
        
        branch_counts = windowed["branch_id"].value_counts(normalize=True)
        patterns[f"{prefix}_branch"] = {k: round(v, 4) for k, v in branch_counts.items()}
        
        bins = [0, 100, 500, 1000, 5000, 10000, 50000, float("inf")]
        labels = ["0-100", "100-500", "500-1k", "1k-5k", "5k-10k", "10k-50k", "50k+"]
        bucketed = pd.cut(windowed["amount"], bins=bins, labels=labels)
        bucket_counts = bucketed.value_counts(normalize=True)
        patterns[f"{prefix}_amount_buckets"] = {k: round(v, 4) for k, v in bucket_counts.items()}
        
        # NEW: client distribution — who does this employee serve?
        client_counts = windowed["client_id"].value_counts(normalize=True)
        patterns[f"{prefix}_client"] = {k: round(v, 4) for k, v in client_counts.items()}
    
    return patterns


def compute_raw_peer_metrics(emp_txs, ref_date, window_days):
    """
    Pass 1: Compute raw metrics for ONE employee at ONE window.
    Returns a dict of raw values (not yet z-scored).
    """
    cutoff = ref_date - timedelta(days=window_days)
    windowed = emp_txs[emp_txs["timestamp"] >= cutoff]
    
    metrics = {}
    
    # Volume — raw count (will be branch-normalized later)
    metrics["volume"] = len(windowed)
    
    # Error/reversal rate — we don't have error flags in our data,
    # so we use a proxy: duplicate transactions (same client, same amount, same day)
    if len(windowed) > 0:
        windowed_copy = windowed.copy()
        windowed_copy["date"] = windowed_copy["timestamp"].dt.date
        dupes = windowed_copy.duplicated(subset=["client_id", "amount", "date"], keep=False)
        metrics["error_rate"] = round(dupes.sum() / len(windowed), 4) if len(windowed) > 0 else 0.0
    else:
        metrics["error_rate"] = 0.0
    
    # Near-threshold ratio — % of transactions in 8,000-9,999 DT band
    if len(windowed) > 0:
        near = windowed[(windowed["amount"] >= NEAR_THRESHOLD_BAND[0]) & 
                        (windowed["amount"] <= NEAR_THRESHOLD_BAND[1])]
        metrics["near_threshold_ratio"] = round(len(near) / len(windowed), 4)
    else:
        metrics["near_threshold_ratio"] = 0.0
    
    # Client concentration — Herfindahl index (sum of squared shares)
    # High = concentrated on few clients, Low = spread across many
    if len(windowed) > 0:
        client_shares = windowed["client_id"].value_counts(normalize=True)
        metrics["client_concentration"] = round(float((client_shares ** 2).sum()), 4)
    else:
        metrics["client_concentration"] = 0.0
    
    # Client-location mismatch — % of transactions where the client's
    # home branch differs from the transaction branch
    # We need client home branch info for this — passed separately
    metrics["client_location_mismatch"] = 0.0  # placeholder, needs client lookup
    
    return metrics


def compute_peer_zscores(all_employee_metrics, employee_branches, branch_avg_volumes):
    """
    Pass 2: Z-score each employee's metrics against the bank-wide population.
    
    Volume is first normalized by branch average, then z-scored.
    All other metrics are z-scored directly (they're already ratios).
    """
    metric_names = ["error_rate", "near_threshold_ratio", 
                    "client_concentration", "client_location_mismatch"]
    
    # --- Volume: normalize by branch first ---
    volume_ratios = {}
    for emp_id, metrics in all_employee_metrics.items():
        branch = employee_branches[emp_id]
        branch_avg = branch_avg_volumes.get(branch, 1)  # avoid division by zero
        volume_ratios[emp_id] = metrics["volume"] / branch_avg if branch_avg > 0 else 0.0
    
    vol_values = list(volume_ratios.values())
    vol_mean = np.mean(vol_values) if vol_values else 0
    vol_std = np.std(vol_values) if vol_values else 1
    
    # --- Z-score everything ---
    zscores = {}
    for emp_id in all_employee_metrics:
        zscores[emp_id] = {}
        
        # Volume z-score (from normalized ratio)
        if vol_std > 0:
            zscores[emp_id]["peer_z_volume"] = round(
                (volume_ratios[emp_id] - vol_mean) / vol_std, 4)
        else:
            zscores[emp_id]["peer_z_volume"] = 0.0
        
        # Other metrics — direct z-score bank-wide
        for metric in metric_names:
            values = [m[metric] for m in all_employee_metrics.values()]
            m_mean = np.mean(values)
            m_std = np.std(values)
            
            if m_std > 0:
                zscores[emp_id][f"peer_z_{metric}"] = round(
                    (all_employee_metrics[emp_id][metric] - m_mean) / m_std, 4)
            else:
                zscores[emp_id][f"peer_z_{metric}"] = 0.0
    
    return zscores

def compute_employee_profile(emp_id, emp_txs, emp_info, ref_date, peer_zscores_30d, peer_zscores_90d):
    profile = {}
    
    # Category 1: Personal info
    profile["branch_id"] = emp_info["branch_id"]
    
    # Category 2: Transaction stats (reused from client logic)
    profile.update(compute_transaction_stats(emp_txs, ref_date))
    
    # Category 3: Behavioral patterns (with client_distribution)
    profile.update(compute_behavioral_patterns(emp_txs, ref_date))
    
    # Category 4: Peer comparison
    if emp_id in peer_zscores_30d:
        for k, v in peer_zscores_30d[emp_id].items():
            profile[f"{k}_30d"] = v
    if emp_id in peer_zscores_90d:
        for k, v in peer_zscores_90d[emp_id].items():
            profile[f"{k}_90d"] = v
    
    # Category 5: Risk history — skipped
    
    return profile


def run_batch(conn):
    print("Fetching transactions...")
    tx_df = fetch_all_transactions(conn)
    print(f"Loaded {len(tx_df)} transactions")
    
    emp_df = pd.read_sql("SELECT * FROM employees_master", conn)
    print(f"Loaded {len(emp_df)} employees")
    
    # Build lookup: employee_id -> branch_id
    employee_branches = dict(zip(emp_df["employee_id"], emp_df["branch_id"]))
    
    grouped = tx_df.groupby("employee_id")
    
    weekly_dates = generate_weekly_dates(SIM_START, SIM_END)
    print(f"Computing employee profiles for {len(weekly_dates)} weekly snapshots...")
    
    total_written = 0
    
    for i, ref_date in enumerate(weekly_dates, 1):
        
        # --- Pass 1: raw metrics for peer comparison ---
        all_metrics_30d = {}
        all_metrics_90d = {}
        
        for _, emp_row in emp_df.iterrows():
            eid = emp_row["employee_id"]
            if eid in grouped.groups:
                all_txs = grouped.get_group(eid)
                emp_txs = all_txs[all_txs["timestamp"] <= ref_date]
            else:
                emp_txs = tx_df.iloc[0:0]
            
            all_metrics_30d[eid] = compute_raw_peer_metrics(emp_txs, ref_date, 30)
            all_metrics_90d[eid] = compute_raw_peer_metrics(emp_txs, ref_date, 90)
        
        # Branch average volumes for normalization
        branch_vols_30d = {}
        branch_vols_90d = {}
        for eid, metrics in all_metrics_30d.items():
            br = employee_branches[eid]
            branch_vols_30d.setdefault(br, []).append(metrics["volume"])
        for eid, metrics in all_metrics_90d.items():
            br = employee_branches[eid]
            branch_vols_90d.setdefault(br, []).append(metrics["volume"])
        
        branch_avg_30d = {br: np.mean(vols) for br, vols in branch_vols_30d.items()}
        branch_avg_90d = {br: np.mean(vols) for br, vols in branch_vols_90d.items()}
        
        # --- Pass 2: z-score ---
        peer_zscores_30d = compute_peer_zscores(all_metrics_30d, employee_branches, branch_avg_30d)
        peer_zscores_90d = compute_peer_zscores(all_metrics_90d, employee_branches, branch_avg_90d)
        
        # --- Assemble profiles ---
        snapshot_profiles = []
        
        for _, emp_row in emp_df.iterrows():
            eid = emp_row["employee_id"]
            if eid in grouped.groups:
                all_txs = grouped.get_group(eid)
                emp_txs = all_txs[all_txs["timestamp"] <= ref_date]
            else:
                emp_txs = tx_df.iloc[0:0]
            
            profile = compute_employee_profile(
                eid, emp_txs, emp_row, ref_date,
                peer_zscores_30d, peer_zscores_90d
            )
            
            snapshot_profiles.append((str(eid), ref_date, json.dumps(profile)))
        
        # Bulk insert
        cur = conn.cursor()
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO employee_profile_snapshots (employee_id, computed_at, profile_data) VALUES %s",
            snapshot_profiles,
            template="(%s, %s, %s::jsonb)"
        )
        conn.commit()
        cur.close()
        
        total_written += len(snapshot_profiles)
        print(f"  [{i}/{len(weekly_dates)}] {ref_date.date()}: {len(snapshot_profiles)} profiles")
    
    print(f"Done — {total_written} employee profile snapshots written.")


if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)
    run_batch(conn)
    conn.close()

