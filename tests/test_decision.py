# tests/test_decision.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score
from datetime import datetime
from src.training.data_prep import prepare
from src.training.autoencoder import Autoencoder
from src.training.lstm import LSTMPredictor, build_sequences
from src.decision.decision import DecisionService, DecisionInput, AlertTier

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ================================================================
# STEP 1: Load everything
# ================================================================
print("Loading data and models...")
data = prepare(str(PROJECT_ROOT / "data" / "enriched_events.csv"))
feature_cols = data["feature_cols"]
scaler = data["scaler"]

ae_ref = np.load(PROJECT_ROOT / "models" / "ae_ref_scores.npy")
lstm_ref = np.load(PROJECT_ROOT / "models" / "lstm_ref_scores.npy")

clients = pd.read_csv(PROJECT_ROOT / "data" / "clients.csv")
sim_start = datetime(2025, 1, 1)
clients["days_since_opening"] = (
    (sim_start - pd.to_datetime(clients["account_opening_date"])).dt.days
)
# After loading clients.csv
clients["expected_daily_rate"] = (
    clients["frequency_retrait"] + clients["frequency_versement"] +
    clients["frequency_virement"] + clients["frequency_cheque"]
) / 30

client_daily_rate = dict(zip(clients["client_id"],
                             clients["expected_daily_rate"]))
client_info = {}
for _, row in clients.iterrows():
    client_info[row["client_id"]] = {
        "days_since_opening": row["days_since_opening"],
        "archetype": row["archetype"],
        "home_branch": row["home_branch"],
    }

# ================================================================
# STEP 2: Score all test events (same as test_fusion.py)
# ================================================================

# AE
ae = Autoencoder(len(feature_cols), hidden_dims=[20, 10])
ae.load_state_dict(torch.load(PROJECT_ROOT / "models" / "autoencoder.pt",
                               weights_only=True))
ae.eval()

X_test_t = torch.FloatTensor(data["X_test"])
with torch.no_grad():
    recon = ae(X_test_t)
    ae_raw = (recon - X_test_t).pow(2).mean(dim=1).numpy()
ae_pct = np.searchsorted(ae_ref, ae_raw, side="right") / len(ae_ref)

# LSTM
with open(PROJECT_ROOT / "models" / "lstm_config.json") as f:
    cfg = json.load(f)
lstm = LSTMPredictor(cfg["input_dim"], cfg["hidden_dim"], cfg["num_layers"])
lstm.load_state_dict(torch.load(PROJECT_ROOT / "models" / "lstm.pt",
                                 weights_only=True))
lstm.eval()

full_df = pd.concat([data["train_df"], data["val_df"], data["test_df"]]) \
            .sort_values("timestamp").reset_index(drop=True)
full_df[feature_cols] = scaler.transform(full_df[feature_cols].values)

sequences, targets, meta = build_sequences(full_df, feature_cols, cfg["seq_len"])
test_seq_mask = (meta["day_offset"] > 153).values
X_seq = torch.FloatTensor(sequences[test_seq_mask])
Y_tgt = torch.FloatTensor(targets[test_seq_mask])
test_meta = meta[test_seq_mask].reset_index(drop=True)

lstm_raw_list = []
for i in range(0, len(X_seq), 512):
    with torch.no_grad():
        pred = lstm(X_seq[i:i+512])
        batch_scores = (pred - Y_tgt[i:i+512]).pow(2).mean(dim=1).numpy()
        lstm_raw_list.append(batch_scores)
lstm_raw = np.concatenate(lstm_raw_list)
lstm_pct = np.searchsorted(lstm_ref, lstm_raw, side="right") / len(lstm_ref)

# ================================================================
# STEP 3: Fuse
# ================================================================
test_df = data["test_df"].copy().reset_index(drop=True)
test_df["ae_pct"] = ae_pct

test_meta["lstm_pct"] = lstm_pct
test_df = test_df.merge(
    test_meta[["event_id", "lstm_pct"]],
    on="event_id",
    how="left"
)

def sigmoid_weight(days, midpoint=90, steepness=15):
    z = (days - midpoint) / steepness
    return 1.0 / (1.0 + np.exp(-z))

test_df["days_since_opening"] = test_df["client_id"].map(
    lambda cid: client_info.get(cid, {}).get("days_since_opening", 365))
test_df["w_lstm"] = sigmoid_weight(test_df["days_since_opening"])

has_lstm = test_df["lstm_pct"].notna()
test_df["fused"] = test_df["ae_pct"]
test_df.loc[has_lstm, "fused"] = (
    (1 - test_df.loc[has_lstm, "w_lstm"]) * test_df.loc[has_lstm, "ae_pct"] +
    test_df.loc[has_lstm, "w_lstm"] * test_df.loc[has_lstm, "lstm_pct"]
)

# ================================================================
# STEP 4: Run Decision Service on every test event
# ================================================================
print("\nRunning Decision Service on test set...")
ds = DecisionService()
decisions = []

# Load raw transactions for amount + operation_type
raw_tx = pd.read_csv(PROJECT_ROOT / "data" / "transactions.csv")
raw_lookup = {}
for _, row in raw_tx.iterrows():
    raw_lookup[row["event_id"]] = row

for idx, row in test_df.iterrows():
    cid = row["client_id"]
    info = client_info.get(cid, {})
    raw = raw_lookup.get(row["event_id"], {})

    # Map enriched feature names to DecisionInput fields
    # Feature availability depends on what's in the enriched CSV
    inp = DecisionInput(
        fused_score=row["fused"],
        ae_pct=row["ae_pct"],
        lstm_pct=row.get("lstm_pct"),
        w_lstm=row["w_lstm"],

        event_id=row["event_id"],
        client_id=cid,
        employee_id=raw.get("employee_id", ""),
        branch_id=raw.get("branch_id", ""),
        operation_type=raw.get("operation_type", ""),
        amount=raw.get("amount", 0),
        timestamp=str(row.get("timestamp", "")),

        # Enriched features (column names from enriched_events.csv)
        z_amount=row.get("z_amount", 0),
        tx_count_24h=int(row.get("tx_count_24h", 0)),
        cumulative_amount_24h=row.get("cumulative_amount_24h", 0),
        near_threshold_count_7d=int(row.get("near_threshold_count_7d", 0)),
        same_employee_client_count_24h=int(row.get("same_employee_client_count_24h", 0)),
        is_new_beneficiary=bool(row.get("is_new_beneficiary", False)),
        is_round_amount=bool(row.get("is_round_amount", False)),
        has_duplicate_recent=bool(row.get("has_duplicate_recent", False)),

        # Client profile
        days_since_opening=info.get("days_since_opening", 365),
        archetype=info.get("archetype", "unknown"),
        home_branch=info.get("home_branch", ""),

        # Risk history — zeros (first run, no history)
        alert_count_30d=0,
        confirmed_count_30d=0,
        rejected_count_30d=0,
        days_since_last_alert=999,
        expected_daily_rate=client_daily_rate.get(cid, 1.0),
    )
    

    decision = ds.decide(inp)
    decisions.append(decision)

# ================================================================
# STEP 5: Analyze results
# ================================================================
test_df["tier"] = [d.tier.name for d in decisions]
test_df["n_reg_flags"] = [len(d.regulatory_flags) for d in decisions]

y_test = test_df["is_anomaly"].values.astype(int)

print(f"\n{'='*60}")
print(f"  DECISION SERVICE RESULTS ({len(test_df)} events)")
print(f"{'='*60}")

# Tier distribution
print(f"\nTier distribution (all events):")
for tier_name in ["INFO", "REVIEW", "ALERT", "BLOCK"]:
    mask = test_df["tier"] == tier_name
    total = mask.sum()
    anomalies = (test_df.loc[mask, "is_anomaly"] == True).sum()
    pct = 100 * total / len(test_df)
    anom_pct = 100 * anomalies / total if total > 0 else 0
    print(f"  {tier_name:<8s}  {total:>6d} ({pct:>5.2f}%)  "
          f"contains {anomalies:>3d} anomalies ({anom_pct:>5.1f}% precision)")

# Daily volumes (test = 27 days)
test_days = 27
print(f"\nDaily alert volumes (avg over {test_days} days):")
for tier_name in ["REVIEW", "ALERT", "BLOCK"]:
    count = (test_df["tier"] == tier_name).sum()
    print(f"  {tier_name:<8s}  {count/test_days:>6.1f} / day")

# Detection rate: what % of anomalies land in REVIEW or above?
above_info = test_df[(test_df["tier"] != "INFO") & (test_df["is_anomaly"] == True)]
total_anomalies = y_test.sum()
print(f"\nAnomaly detection:")
print(f"  Total anomalies in test set: {total_anomalies}")
print(f"  Caught (REVIEW+ALERT+BLOCK): {len(above_info)} "
      f"({100*len(above_info)/total_anomalies:.1f}%)")

# Missed anomalies — what scenarios slip through?
missed = test_df[(test_df["tier"] == "INFO") & (test_df["is_anomaly"] == True)]
if len(missed) > 0:
    print(f"  Missed (still INFO): {len(missed)}")
    print(f"\n  Missed by scenario:")
    for scenario, count in missed["anomaly_type"].value_counts().items():
        print(f"    {scenario:<30s}  {count}")

# Regulatory flag impact
reg_flagged = test_df[test_df["n_reg_flags"] > 0]
print(f"\nRegulatory flags:")
print(f"  Events with regulatory flags: {len(reg_flagged)}")
print(f"  Of which anomalies: {reg_flagged['is_anomaly'].sum()}")

# Show a few example decisions
print(f"\n{'='*60}")
print(f"  EXAMPLE DECISIONS (5 anomalies)")
print(f"{'='*60}")
anomaly_decisions = [d for d, r in zip(decisions, y_test) if r == 1]
for d in anomaly_decisions[:5]:
    print(f"\n  Event: {d.event_id[:12]}...")
    print(f"  Tier:  {d.tier.name}  |  Score: {d.fused_score:.4f}")
    for r in d.reasons:
        print(f"    → {r}")
    for f in d.regulatory_flags:
        print(f"    ⚠ {f}")