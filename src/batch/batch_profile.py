
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
    "password": "9011361923257228",
}

# Reference point for window calculations.
# In production this would be datetime.now().
# For our simulation (180 days from 2025-01-01) we use the last day.
REFERENCE_DATE = datetime(2025, 6, 29)

WINDOWS = {
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "90d": timedelta(days=90),
}

OPERATIONS = ["retrait", "versement", "virement", "cheque"]


def fetch_all_transactions(conn):
    """Load all transactions into a DataFrame. 
    For 340k rows this fits comfortably in memory."""
    query = "SELECT * FROM transactions"
    df = pd.read_sql(query, conn, parse_dates=["timestamp"])
    return df


def compute_transaction_stats(client_txs, ref_date):
    """
    Category 2: Transaction stats per operation × window.
    
    For each (operation, window) pair, computes:
      - count, sum, mean, std, min, max
    
    This produces 4 ops × 4 windows × 6 stats = 96 fields.
    """
    stats = {}
    
    for window_name, window_delta in WINDOWS.items():
        cutoff = ref_date - window_delta
        windowed = client_txs[client_txs["timestamp"] >= cutoff]
        
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


def compute_behavioral_patterns(client_txs, ref_date):
        """
        Category 3: Behavioral patterns.
        
        5 distributions at 30d and 180d windows:
        - amount_distribution (histogram buckets)
        - operation_distribution (% per op type)
        - hour_distribution (% per hour bucket)
        - day_of_week_distribution (% per weekday)
        - branch_distribution (% per branch visited)
        
        Plus known counterparty sets.
        """
        patterns = {}
        
        for window_name, days in [("30d", 30), ("180d", 180)]:
            cutoff = ref_date - timedelta(days=days)
            windowed = client_txs[client_txs["timestamp"] >= cutoff]
            
            prefix = f"dist_{window_name}"
            
            if len(windowed) == 0:
                patterns[f"{prefix}_operation"] = {}
                patterns[f"{prefix}_hour"] = {}
                patterns[f"{prefix}_day_of_week"] = {}
                patterns[f"{prefix}_branch"] = {}
                patterns[f"{prefix}_amount_buckets"] = {}
                continue
            
            # Operation distribution — what % of transactions is each op type
            op_counts = windowed["operation_type"].value_counts(normalize=True)
            patterns[f"{prefix}_operation"] = {k: round(v, 4) for k, v in op_counts.items()}
            
            # Hour distribution — when during the day do they transact
            hours = windowed["timestamp"].dt.hour
            hour_counts = hours.value_counts(normalize=True).sort_index()
            patterns[f"{prefix}_hour"] = {str(k): round(v, 4) for k, v in hour_counts.items()}
            
            # Day of week distribution
            dow = windowed["timestamp"].dt.dayofweek  # 0=Mon, 6=Sun
            dow_counts = dow.value_counts(normalize=True).sort_index()
            patterns[f"{prefix}_day_of_week"] = {str(k): round(v, 4) for k, v in dow_counts.items()}
            
            # Branch distribution — how spread out across branches
            branch_counts = windowed["branch_id"].value_counts(normalize=True)
            patterns[f"{prefix}_branch"] = {k: round(v, 4) for k, v in branch_counts.items()}
            
            # Amount buckets — histogram of transaction amounts
            bins = [0, 100, 500, 1000, 5000, 10000, 50000, float("inf")]
            labels = ["0-100", "100-500", "500-1k", "1k-5k", "5k-10k", "10k-50k", "50k+"]
            bucketed = pd.cut(windowed["amount"], bins=bins, labels=labels)
            bucket_counts = bucketed.value_counts(normalize=True)
            patterns[f"{prefix}_amount_buckets"] = {k: round(v, 4) for k, v in bucket_counts.items()}
        
        # Known counterparties over 365 days (full simulation)
        cutoff_365 = ref_date - timedelta(days=365)
        year_txs = client_txs[client_txs["timestamp"] >= cutoff_365]
        
        # Extract beneficiaries from virement payloads
        virement_txs = year_txs[year_txs["operation_type"] == "virement"]
        beneficiaries = set()
        for _, row in virement_txs.iterrows():
            payload = row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"])
            bid = payload.get("beneficiary_id")
            if bid and bid.startswith("KNOWN"):
                beneficiaries.add(bid)
        
        # Extract emitters from cheque payloads
        cheque_txs = year_txs[year_txs["operation_type"] == "cheque"]
        emitters = set()
        for _, row in cheque_txs.iterrows():
            payload = row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"])
            eid = payload.get("emitter_id")
            if eid and eid.startswith("KNOWN"):
                emitters.add(eid)
        
        patterns["known_beneficiaries_365d"] = list(beneficiaries)
        patterns["known_beneficiaries_count"] = len(beneficiaries)
        patterns["known_emitters_365d"] = list(emitters)
        patterns["known_emitters_count"] = len(emitters)
        
        return patterns


def compute_financial_state(client_txs, ref_date):
    """
    Category 4: Financial state.
    
    We don't have real balance data, but we can compute
    cumulative flows from transactions. These are proxies.
    """
    state = {}
    
    for window_name, days in [("30d", 30), ("90d", 90)]:
        cutoff = ref_date - timedelta(days=days)
        windowed = client_txs[client_txs["timestamp"] >= cutoff]
        
        prefix = f"flow_{window_name}"
        
        # Inflows: versement + cheque (money coming in)
        inflow = windowed[windowed["operation_type"].isin(["versement", "cheque"])]["amount"].sum()
        
        # Outflows: retrait + virement (money going out)
        outflow = windowed[windowed["operation_type"].isin(["retrait", "virement"])]["amount"].sum()
        
        state[f"{prefix}_inflow"] = round(float(inflow), 2)
        state[f"{prefix}_outflow"] = round(float(outflow), 2)
        state[f"{prefix}_net"] = round(float(inflow - outflow), 2)
    
    return state


def compute_recent_events_buffer(client_txs, n=50):
    """
    Category 6: Last N events for LSTM sequence input.
    
    Returns a list of simplified event dicts, ordered chronologically.
    The LSTM consumes this as its input sequence.
    """
    recent = client_txs.nlargest(n, "timestamp").sort_values("timestamp")
    
    buffer = []
    for _, row in recent.iterrows():
        buffer.append({
            "timestamp": row["timestamp"].isoformat(),
            "operation_type": row["operation_type"],
            "amount": float(row["amount"]),
            "branch_id": row["branch_id"],
            "employee_id": row["employee_id"],
        })
    
    return buffer


def compute_client_profile(client_id, client_txs, client_info, ref_date):
    profile = {}
    
    # Category 1: Personal info
    profile["archetype"] = client_info["archetype"]
    profile["client_type"] = client_info["client_type"]
    profile["home_branch_id"] = client_info["home_branch_id"]
    profile["account_age_days"] = (ref_date.date() - client_info["account_opening_date"].date()).days
    profile["maturity_status"] = "mature" if profile["account_age_days"] > 90 else "new"
    
    # Category 2: Transaction stats
    profile.update(compute_transaction_stats(client_txs, ref_date))
    
    # Category 3: Behavioral patterns
    profile.update(compute_behavioral_patterns(client_txs, ref_date))
    
    # Category 4: Financial state
    profile.update(compute_financial_state(client_txs, ref_date))
    
    # Category 5: Risk history — skipped, no alerts yet
    
    # Category 6: Recent events buffer
    profile["recent_events_buffer"] = compute_recent_events_buffer(client_txs)
    
    return profile


def run_batch(conn):
    print("Fetching transactions...")
    tx_df = fetch_all_transactions(conn)
    print(f"Loaded {len(tx_df)} transactions")
    
    # Fetch client master data
    clients_df = pd.read_sql("SELECT * FROM clients_master", conn, 
                              parse_dates=["account_opening_date"])
    print(f"Loaded {len(clients_df)} clients")
    
    # Group transactions by client
    grouped = tx_df.groupby("client_id")
    
    profiles = []
    for _, client_row in clients_df.iterrows():
        cid = client_row["client_id"]
        
        # Get this client's transactions (empty DataFrame if none)
        if cid in grouped.groups:
            client_txs = grouped.get_group(cid)
        else:
            client_txs = tx_df.iloc[0:0]  # empty with same columns
        
        profile = compute_client_profile(
            cid, client_txs, client_row, REFERENCE_DATE
        )
        
        profiles.append((str(cid), REFERENCE_DATE, json.dumps(profile)))
    
    # Bulk insert into profile_snapshots
    print(f"Writing {len(profiles)} profiles...")
    cur = conn.cursor()
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO profile_snapshots (client_id, computed_at, profile_data) VALUES %s",
        profiles,
        template="(%s, %s, %s::jsonb)"
    )
    conn.commit()
    cur.close()
    print("Done.")


if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)
    run_batch(conn)
    conn.close()