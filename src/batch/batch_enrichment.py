import psycopg2
import pandas as pd
import numpy as np
import json
import uuid
from datetime import datetime, timedelta

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "amen_anomaly",
    "user": "postgres",
    "password": "pwd",
}

OPERATIONS = ["retrait", "versement", "virement", "cheque"]


def load_data(conn):
    """Load all required data from PostgreSQL."""

    tx_df = pd.read_sql("SELECT * FROM transactions", conn, parse_dates=["timestamp"])
    tx_df = tx_df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(tx_df)} transactions")

    # Client profiles — max_level=0 keeps nested dicts as dicts
    prof_df = pd.read_sql(
        "SELECT client_id, computed_at, profile_data FROM profile_snapshots",
        conn, parse_dates=["computed_at"])
    prof_expanded = pd.json_normalize(prof_df["profile_data"], max_level=0)
    prof_df = pd.concat([prof_df[["client_id", "computed_at"]], prof_expanded], axis=1)
    prof_df = prof_df.sort_values("computed_at").reset_index(drop=True)
    print(f"Loaded {len(prof_df)} client profile snapshots ({len(prof_expanded.columns)} fields)")

    # Employee profiles
    emp_prof_df = pd.read_sql(
        "SELECT employee_id, computed_at, profile_data FROM employee_profile_snapshots",
        conn, parse_dates=["computed_at"])
    emp_expanded = pd.json_normalize(emp_prof_df["profile_data"], max_level=0)
    emp_prof_df = pd.concat([emp_prof_df[["employee_id", "computed_at"]], emp_expanded], axis=1)
    emp_prof_df = emp_prof_df.sort_values("computed_at").reset_index(drop=True)
    print(f"Loaded {len(emp_prof_df)} employee profile snapshots")

    # Archetype baselines
    baselines_df = pd.read_sql("SELECT * FROM archetype_baselines", conn)
    baselines = {}
    for _, row in baselines_df.iterrows():
        data = row["baseline_data"]
        baselines[row["archetype"]] = data if isinstance(data, dict) else json.loads(data)
    print(f"Loaded {len(baselines)} archetype baselines")

    # Client master
    clients_df = pd.read_sql(
        "SELECT client_id, home_branch_id, initial_balance FROM clients_master", conn)
    print(f"Loaded {len(clients_df)} clients")

    return tx_df, prof_df, emp_prof_df, baselines, clients_df


def merge_profiles(tx_df, prof_df, emp_prof_df):
    """As-of join: match each transaction to the most recent profile snapshot."""

    merged = pd.merge_asof(
        tx_df,
        prof_df,
        left_on="timestamp",
        right_on="computed_at",
        by="client_id",
        direction="backward",
        suffixes=("", "_prof")
    )

    merged = pd.merge_asof(
        merged.sort_values("timestamp"),
        emp_prof_df,
        left_on="timestamp",
        right_on="computed_at",
        by="employee_id",
        direction="backward",
        suffixes=("", "_emp")
    )

    print(f"Merged: {len(merged)} rows")
    return merged


# ============================================================
# Helper: safe dict/list lookups for profile columns that may be NaN
# ============================================================

def safe_dict_get(d, key, default=0.0):
    """Get a value from a dict, returning default if d is not a dict."""
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def safe_list_contains(lst, item):
    """Check if item is in lst, returning False if lst is not a list."""
    if isinstance(lst, list):
        return item in lst
    return False


# ============================================================
# Section 1: Metadata & Traceability
# ============================================================

def enrich_section_1_metadata(df):
    df["enriched_event_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df["raw_event_id"] = df["event_id"]
    df["enrichment_timestamp"] = datetime.now().isoformat()
    df["raw_event_timestamp"] = df["timestamp"]
    df["client_profile_snapshot_at"] = df["computed_at"]
    df["employee_profile_snapshot_at"] = df.get("computed_at_emp")
    return df


# ============================================================
# Section 3: Cold-Start & Maturity Flags
# ============================================================

def enrich_section_3_coldstart(df):
    df["is_client_mature"] = df["maturity_status"].apply(
        lambda v: v == "mature" if isinstance(v, str) else False
    )
    df["client_archetype"] = df["archetype"].apply(
        lambda v: v if isinstance(v, str) else "unknown"
    )
    df["is_employee_established"] = True
    return df


# ============================================================
# Section 4: Regulatory Rule Signals
# ============================================================

def enrich_section_4_regulatory(df):
    df["is_above_threshold"] = df["amount"] >= 10000
    df["is_near_threshold_band"] = (df["amount"] >= 8000) & (df["amount"] < 10000)

    hour = df["timestamp"].dt.hour
    df["is_outside_business_hours"] = (hour < 8) | (hour >= 16)

    dow = df["timestamp"].dt.dayofweek
    df["is_outside_business_days"] = dow >= 5

    df["_date"] = df["timestamp"].dt.date
    dupes = df.duplicated(subset=["client_id", "amount", "_date"], keep=False)
    df["has_duplicate_recent"] = dupes
    df.drop(columns=["_date"], inplace=True)

    return df


# ============================================================
# Section 5: Client-Side Derived Features
# ============================================================

def enrich_section_5_client_features(df, clients_df):

    # --- 5a: Z-scores ---
    df["z_amount_30d"] = 0.0
    df["z_amount_90d"] = 0.0

    for op in OPERATIONS:
        mask = df["operation_type"] == op

        mean_col_30 = f"tx_{op}_30d_mean"
        std_col_30 = f"tx_{op}_30d_std"
        mean_col_90 = f"tx_{op}_90d_mean"
        std_col_90 = f"tx_{op}_90d_std"

        if mean_col_30 in df.columns and std_col_30 in df.columns:
            mean_30 = pd.to_numeric(df.loc[mask, mean_col_30], errors="coerce").fillna(0)
            std_30 = pd.to_numeric(df.loc[mask, std_col_30], errors="coerce").fillna(0).replace(0, 1)
            df.loc[mask, "z_amount_30d"] = ((df.loc[mask, "amount"] - mean_30) / std_30).round(4)

        if mean_col_90 in df.columns and std_col_90 in df.columns:
            mean_90 = pd.to_numeric(df.loc[mask, mean_col_90], errors="coerce").fillna(0)
            std_90 = pd.to_numeric(df.loc[mask, std_col_90], errors="coerce").fillna(0).replace(0, 1)
            df.loc[mask, "z_amount_90d"] = ((df.loc[mask, "amount"] - mean_90) / std_90).round(4)

    # --- 5b: Frequency lookups ---
    hour_str = df["timestamp"].dt.hour.astype(str)
    dow_str = df["timestamp"].dt.dayofweek.astype(str)

    df["branch_frequency_30d"] = df.apply(
        lambda r: safe_dict_get(r.get("dist_30d_branch"), r["branch_id"]), axis=1)
    df["branch_frequency_180d"] = df.apply(
        lambda r: safe_dict_get(r.get("dist_180d_branch"), r["branch_id"]), axis=1)

    df["_hour_str"] = hour_str
    df["hour_frequency_30d"] = df.apply(
        lambda r: safe_dict_get(r.get("dist_30d_hour"), r["_hour_str"]), axis=1)
    df["hour_frequency_180d"] = df.apply(
        lambda r: safe_dict_get(r.get("dist_180d_hour"), r["_hour_str"]), axis=1)

    df["_dow_str"] = dow_str
    df["dow_frequency_30d"] = df.apply(
        lambda r: safe_dict_get(r.get("dist_30d_day_of_week"), r["_dow_str"]), axis=1)
    df["dow_frequency_180d"] = df.apply(
        lambda r: safe_dict_get(r.get("dist_180d_day_of_week"), r["_dow_str"]), axis=1)

    df["employee_frequency_30d"] = 0.0
    df["employee_frequency_180d"] = 0.0

    df.drop(columns=["_hour_str", "_dow_str"], inplace=True)

    # --- 5c: Counterparty features ---
    df["_payload"] = df["payload"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else (x if isinstance(x, dict) else {}))

    df["_beneficiary"] = df["_payload"].apply(lambda p: p.get("beneficiary_id"))
    df["_emitter"] = df["_payload"].apply(lambda p: p.get("emitter_id"))

    df["is_first_time_beneficiary"] = df.apply(
        lambda r: (r["_beneficiary"] is not None and
                   not safe_list_contains(r.get("known_beneficiaries_365d"), r["_beneficiary"])),
        axis=1)

    df["is_first_time_emitter"] = df.apply(
        lambda r: (r["_emitter"] is not None and
                   not safe_list_contains(r.get("known_emitters_365d"), r["_emitter"])),
        axis=1)

    df["days_since_first_seen_beneficiary"] = 0
    df["days_since_first_seen_emitter"] = 0

    df.drop(columns=["_payload", "_beneficiary", "_emitter"], inplace=True)

    # --- 5d: Cumulative amounts ---
    df = df.sort_values(["client_id", "timestamp"]).reset_index(drop=True)

    cumulative_24h = []
    cumulative_7d = []

    for cid, group in df.groupby("client_id"):
        group_sorted = group.sort_values("timestamp")
        ts = group_sorted["timestamp"].values
        amounts = group_sorted["amount"].values

        cum_24 = []
        cum_7 = []
        for i in range(len(group_sorted)):
            t = ts[i]
            mask_24 = (ts[:i] >= t - np.timedelta64(24, 'h')) & (ts[:i] < t)
            cum_24.append(float(amounts[:i][mask_24].sum()))

            mask_7 = (ts[:i] >= t - np.timedelta64(7, 'D')) & (ts[:i] < t)
            cum_7.append(float(amounts[:i][mask_7].sum()))

        cumulative_24h.extend(cum_24)
        cumulative_7d.extend(cum_7)

    df["cumulative_amount_24h"] = cumulative_24h
    df["cumulative_amount_7d"] = cumulative_7d

    df["would_cross_threshold_24h"] = (df["cumulative_amount_24h"] + df["amount"]) >= 10000

    total_24h = df["cumulative_amount_24h"] + df["amount"]
    df["near_threshold_ratio_24h"] = (total_24h >= 8000) & (total_24h < 10000)

    # --- 5e: Balance features ---
    df = df.merge(clients_df[["client_id", "initial_balance"]], on="client_id", how="left")

    is_inflow = df["operation_type"].isin(["versement", "cheque"])
    df["_signed_amount"] = np.where(is_inflow, df["amount"], -df["amount"])

    df["_cum_flow"] = df.groupby("client_id")["_signed_amount"].cumsum() - df["_signed_amount"]
    df["current_balance"] = df["initial_balance"].fillna(0) + df["_cum_flow"]

    df["amount_vs_balance_ratio"] = np.where(
        df["current_balance"] > 0,
        (df["amount"] / df["current_balance"]).round(4),
        0.0)

    df["z_balance_30d"] = 0.0

    flow_outflow = pd.to_numeric(df.get("flow_30d_outflow"), errors="coerce").fillna(0)
    flow_inflow = pd.to_numeric(df.get("flow_30d_inflow"), errors="coerce").fillna(0)
    df["inflow_vs_outflow_ratio_30d"] = np.where(
        flow_outflow > 0,
        (flow_inflow / flow_outflow).round(4),
        0.0)

    df.drop(columns=["_signed_amount", "_cum_flow", "initial_balance"], inplace=True, errors="ignore")

    return df


# ============================================================
# Section 6: Archetype Baseline Features
# ============================================================

def enrich_section_6_archetype(df, baselines):
    df["z_amount_vs_archetype"] = 0.0

    for archetype, baseline in baselines.items():
        for op in OPERATIONS:
            mask = (df["client_archetype"] == archetype) & (df["operation_type"] == op)
            mean = baseline.get(f"amount_{op}_mean", 0)
            std = baseline.get(f"amount_{op}_std", 1)
            if std == 0:
                std = 1
            df.loc[mask, "z_amount_vs_archetype"] = ((df.loc[mask, "amount"] - mean) / std).round(4)

    df["_hour_str"] = df["timestamp"].dt.hour.astype(str)
    df["_dow_str"] = df["timestamp"].dt.dayofweek.astype(str)

    df["branch_frequency_vs_archetype"] = df.apply(
        lambda r: safe_dict_get(
            baselines.get(r["client_archetype"], {}).get("branch_distribution"),
            r["branch_id"]),
        axis=1)

    df["hour_frequency_vs_archetype"] = df.apply(
        lambda r: safe_dict_get(
            baselines.get(r["client_archetype"], {}).get("hour_distribution"),
            r["_hour_str"]),
        axis=1)

    df["dow_frequency_vs_archetype"] = df.apply(
        lambda r: safe_dict_get(
            baselines.get(r["client_archetype"], {}).get("dow_distribution"),
            r["_dow_str"]),
        axis=1)

    df.drop(columns=["_hour_str", "_dow_str"], inplace=True)

    return df


# ============================================================
# Section 7: Behavioral Drift Features
# ============================================================

def js_divergence(p_dict, q_dict):
    """Jensen-Shannon divergence between two distributions (as dicts)."""
    if not isinstance(p_dict, dict) or not isinstance(q_dict, dict):
        return 0.0

    all_keys = set(list(p_dict.keys()) + list(q_dict.keys()))
    if not all_keys:
        return 0.0

    p = np.array([float(p_dict.get(k, 0.0)) for k in all_keys])
    q = np.array([float(q_dict.get(k, 0.0)) for k in all_keys])

    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    return round(float(0.5 * kl_pm + 0.5 * kl_qm), 6)


def enrich_section_7_drift(df):
    drift_pairs = [
        ("branch",    "dist_30d_branch",       "dist_180d_branch"),
        ("hour",      "dist_30d_hour",          "dist_180d_hour"),
        ("dow",       "dist_30d_day_of_week",   "dist_180d_day_of_week"),
        ("operation", "dist_30d_operation",      "dist_180d_operation"),
    ]

    for name, col_30, col_180 in drift_pairs:
        df[f"{name}_drift_30d_vs_180d"] = df.apply(
            lambda r, c30=col_30, c180=col_180: js_divergence(r.get(c30), r.get(c180)),
            axis=1)

    # Employee drift — from employee profile's client distribution
    df["employee_drift_30d_vs_180d"] = df.apply(
        lambda r: js_divergence(r.get("dist_30d_client"), r.get("dist_180d_client")),
        axis=1)

    return df


# ============================================================
# Section 8: Employee-Side Derived Features
# ============================================================

def enrich_section_8_employee(df):
    df["z_amount_employee_30d"] = 0.0
    df["z_amount_employee_90d"] = 0.0

    for op in OPERATIONS:
        mask = df["operation_type"] == op

        mean_col = f"tx_{op}_30d_mean_emp"
        std_col = f"tx_{op}_30d_std_emp"
        if mean_col in df.columns and std_col in df.columns:
            mean_30 = pd.to_numeric(df.loc[mask, mean_col], errors="coerce").fillna(0)
            std_30 = pd.to_numeric(df.loc[mask, std_col], errors="coerce").fillna(0).replace(0, 1)
            df.loc[mask, "z_amount_employee_30d"] = ((df.loc[mask, "amount"] - mean_30) / std_30).round(4)

        mean_col_90 = f"tx_{op}_90d_mean_emp"
        std_col_90 = f"tx_{op}_90d_std_emp"
        if mean_col_90 in df.columns and std_col_90 in df.columns:
            mean_90 = pd.to_numeric(df.loc[mask, mean_col_90], errors="coerce").fillna(0)
            std_90 = pd.to_numeric(df.loc[mask, std_col_90], errors="coerce").fillna(0).replace(0, 1)
            df.loc[mask, "z_amount_employee_90d"] = ((df.loc[mask, "amount"] - mean_90) / std_90).round(4)

    df["_hour_str"] = df["timestamp"].dt.hour.astype(str)

    df["branch_frequency_employee_30d"] = df.apply(
        lambda r: safe_dict_get(r.get("dist_30d_branch_emp"), r["branch_id"]), axis=1)
    df["branch_frequency_employee_180d"] = df.apply(
        lambda r: safe_dict_get(r.get("dist_180d_branch_emp"), r["branch_id"]), axis=1)

    df["hour_frequency_employee_30d"] = df.apply(
        lambda r: safe_dict_get(r.get("dist_30d_hour_emp"), r["_hour_str"]), axis=1)
    df["hour_frequency_employee_180d"] = df.apply(
        lambda r: safe_dict_get(r.get("dist_180d_hour_emp"), r["_hour_str"]), axis=1)

    df["client_frequency_employee_30d"] = df.apply(
        lambda r: safe_dict_get(r.get("dist_30d_client"), r["client_id"]), axis=1)
    df["client_frequency_employee_180d"] = df.apply(
        lambda r: safe_dict_get(r.get("dist_180d_client"), r["client_id"]), axis=1)

    df["client_location_mismatch"] = df["home_branch_id"] != df["branch_id"]

    df.drop(columns=["_hour_str"], inplace=True)

    return df


# ============================================================
# Section 9: Peer Comparison (passthrough from employee profile)
# ============================================================

def enrich_section_9_peer(df):
    peer_fields = [
        "peer_z_volume", "peer_z_error_rate", "peer_z_near_threshold_ratio",
        "peer_z_client_concentration", "peer_z_client_location_mismatch"
    ]

    for field in peer_fields:
        for window in ["30d", "90d"]:
            col = f"{field}_{window}"
            # Try _emp suffix first (from merge), then without
            emp_col = f"{col}_emp"
            if emp_col in df.columns:
                df[col] = pd.to_numeric(df[emp_col], errors="coerce").fillna(0.0)
            elif col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                df[col] = 0.0

    return df


# ============================================================
# Section 10: Risk History (all zeros — no alerts yet)
# ============================================================

def enrich_section_10_risk_history(df):
    df["client_recent_alert_count_7d"] = 0
    df["client_confirmed_alert_count_30d"] = 0
    df["client_days_since_last_alert"] = -1
    df["employee_recent_alert_count_7d"] = 0
    df["employee_confirmed_alert_count_30d"] = 0
    df["employee_days_since_last_alert"] = -1
    return df


# ============================================================
# Section 11: LSTM Sequence Input
# ============================================================

def enrich_section_11_lstm(df):
    if "recent_events_buffer" in df.columns:
        df["recent_events_sequence"] = df["recent_events_buffer"].apply(
            lambda v: v if isinstance(v, list) else [])
    else:
        df["recent_events_sequence"] = None
    return df


# ============================================================
# Output column selection
# ============================================================

def select_output_columns(df):
    output_cols = [
        # Identifiers
        "enriched_event_id", "raw_event_id", "client_id", "employee_id", "branch_id",
        "raw_event_timestamp", "operation_type", "amount",
        "client_profile_snapshot_at", "employee_profile_snapshot_at",

        # Section 3
        "is_client_mature", "is_employee_established", "client_archetype",

        # Section 4
        "is_above_threshold", "is_near_threshold_band",
        "is_outside_business_hours", "is_outside_business_days", "has_duplicate_recent",

        # Section 5
        "z_amount_30d", "z_amount_90d",
        "branch_frequency_30d", "branch_frequency_180d",
        "hour_frequency_30d", "hour_frequency_180d",
        "dow_frequency_30d", "dow_frequency_180d",
        "is_first_time_beneficiary", "is_first_time_emitter",
        "days_since_first_seen_beneficiary", "days_since_first_seen_emitter",
        "cumulative_amount_24h", "cumulative_amount_7d",
        "would_cross_threshold_24h", "near_threshold_ratio_24h",
        "current_balance", "amount_vs_balance_ratio",
        "z_balance_30d", "inflow_vs_outflow_ratio_30d",

        # Section 6
        "z_amount_vs_archetype",
        "branch_frequency_vs_archetype", "hour_frequency_vs_archetype",
        "dow_frequency_vs_archetype",

        # Section 7
        "branch_drift_30d_vs_180d", "hour_drift_30d_vs_180d",
        "dow_drift_30d_vs_180d", "operation_drift_30d_vs_180d",
        "employee_drift_30d_vs_180d",

        # Section 8
        "z_amount_employee_30d", "z_amount_employee_90d",
        "branch_frequency_employee_30d", "branch_frequency_employee_180d",
        "hour_frequency_employee_30d", "hour_frequency_employee_180d",
        "client_frequency_employee_30d", "client_frequency_employee_180d",
        "client_location_mismatch",

        # Section 9
        "peer_z_volume_30d", "peer_z_volume_90d",
        "peer_z_error_rate_30d", "peer_z_error_rate_90d",
        "peer_z_near_threshold_ratio_30d", "peer_z_near_threshold_ratio_90d",
        "peer_z_client_concentration_30d", "peer_z_client_concentration_90d",
        "peer_z_client_location_mismatch_30d", "peer_z_client_location_mismatch_90d",

        # Section 10
        "client_recent_alert_count_7d", "client_confirmed_alert_count_30d",
        "client_days_since_last_alert",
        "employee_recent_alert_count_7d", "employee_confirmed_alert_count_30d",
        "employee_days_since_last_alert",

        # Labels
        "is_anomaly", "anomaly_type",
    ]

    existing = [c for c in output_cols if c in df.columns]
    return df[existing]


# ============================================================
# Main orchestrator
# ============================================================

def run_enrichment():
    conn = psycopg2.connect(**DB_CONFIG)

    print("Loading data...")
    tx_df, prof_df, emp_prof_df, baselines, clients_df = load_data(conn)
    conn.close()

    print("Merging profiles...")
    df = merge_profiles(tx_df, prof_df, emp_prof_df)

    # Merge home_branch for location mismatch
    # Merge home_branch for location mismatch — only if not already from profile
    if "home_branch_id" not in df.columns:
        df = df.merge(clients_df[["client_id", "home_branch_id"]], on="client_id", how="left")

    print("Enriching section 1: Metadata...")
    df = enrich_section_1_metadata(df)

    print("Enriching section 3: Cold-start flags...")
    df = enrich_section_3_coldstart(df)

    print("Enriching section 4: Regulatory signals...")
    df = enrich_section_4_regulatory(df)

    print("Enriching section 5: Client features...")
    df = enrich_section_5_client_features(df, clients_df)

    print("Enriching section 6: Archetype baselines...")
    df = enrich_section_6_archetype(df, baselines)

    print("Enriching section 7: Behavioral drift...")
    df = enrich_section_7_drift(df)

    print("Enriching section 8: Employee features...")
    df = enrich_section_8_employee(df)

    print("Enriching section 9: Peer comparison...")
    df = enrich_section_9_peer(df)

    print("Enriching section 10: Risk history...")
    df = enrich_section_10_risk_history(df)

    print("Enriching section 11: LSTM buffer...")
    df = enrich_section_11_lstm(df)

    print("Selecting output columns...")
    output = select_output_columns(df)

    output_path = "data/enriched_events.csv"
    output.to_csv(output_path, index=False)
    print(f"Done — {len(output)} enriched events written to {output_path}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    run_enrichment()