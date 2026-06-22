# scripts/calibrate_thresholds.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from src.training.data_prep import prepare
from src.training.autoencoder import Autoencoder
from src.training.lstm import LSTMPredictor, build_sequences
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Load ────────────────────────────────────────────────────
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
client_maturity = dict(zip(clients["client_id"],
                           clients["days_since_opening"]))

# ── AE on validation set ───────────────────────────────────
ae = Autoencoder(len(feature_cols), hidden_dims=[20, 10])
ae.load_state_dict(torch.load(PROJECT_ROOT / "models" / "autoencoder.pt",
                               weights_only=True))
ae.eval()

X_val_t = torch.FloatTensor(data["X_val"])
with torch.no_grad():
    recon = ae(X_val_t)
    ae_raw = (recon - X_val_t).pow(2).mean(dim=1).numpy()

ae_pct = np.searchsorted(ae_ref, ae_raw, side="right") / len(ae_ref)

# ── LSTM on validation set ─────────────────────────────────
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

val_seq_mask = ((meta["day_offset"] > 126) & (meta["day_offset"] <= 153)).values
X_seq = torch.FloatTensor(sequences[val_seq_mask])
Y_tgt = torch.FloatTensor(targets[val_seq_mask])
val_meta = meta[val_seq_mask].reset_index(drop=True)

lstm_raw_list = []
for i in range(0, len(X_seq), 512):
    with torch.no_grad():
        pred = lstm(X_seq[i:i+512])
        batch_scores = (pred - Y_tgt[i:i+512]).pow(2).mean(dim=1).numpy()
        lstm_raw_list.append(batch_scores)
lstm_raw = np.concatenate(lstm_raw_list)
lstm_pct = np.searchsorted(lstm_ref, lstm_raw, side="right") / len(lstm_ref)

# ── Fuse ────────────────────────────────────────────────────
val_df = data["val_df"].copy().reset_index(drop=True)
val_df["ae_pct"] = ae_pct

val_meta["lstm_pct"] = lstm_pct
val_df = val_df.merge(
    val_meta[["event_id", "lstm_pct"]],
    on="event_id",
    how="left"
)

val_df["days_since_opening"] = val_df["client_id"].map(client_maturity)

def sigmoid_weight(days, midpoint=90, steepness=15):
    z = (days - midpoint) / steepness
    return 1.0 / (1.0 + np.exp(-z))

val_df["w_lstm"] = sigmoid_weight(val_df["days_since_opening"])

has_lstm = val_df["lstm_pct"].notna()
val_df["fused"] = val_df["ae_pct"]
val_df.loc[has_lstm, "fused"] = (
    (1 - val_df.loc[has_lstm, "w_lstm"]) * val_df.loc[has_lstm, "ae_pct"] +
    val_df.loc[has_lstm, "w_lstm"] * val_df.loc[has_lstm, "lstm_pct"]
)

# ── Analyze distributions ───────────────────────────────────
y_val = val_df["is_anomaly"].values.astype(int)
normal_scores = val_df.loc[y_val == 0, "fused"].values
anomaly_scores = val_df.loc[y_val == 1, "fused"].values

print(f"\n{'='*55}")
print(f"  VALIDATION SET FUSED SCORE DISTRIBUTIONS")
print(f"{'='*55}")

print(f"\nNormal events ({len(normal_scores)}):")
for p in [50, 75, 90, 95, 97, 99, 99.5]:
    print(f"  p{p:>5.1f}: {np.percentile(normal_scores, p):.4f}")

print(f"\nAnomaly events ({len(anomaly_scores)}):")
for p in [1, 5, 10, 25, 50, 75, 90]:
    print(f"  p{p:>5.1f}: {np.percentile(anomaly_scores, p):.4f}")

# ── Threshold candidates ────────────────────────────────────
print(f"\n{'='*55}")
print(f"  THRESHOLD ANALYSIS")
print(f"{'='*55}")

candidates = [0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
print(f"\n  {'Threshold':>10s}  {'Normal flagged':>15s}  {'Anomaly caught':>15s}  {'Precision':>10s}")
print(f"  {'-'*55}")

for t in candidates:
    normal_flagged = (normal_scores >= t).sum()
    anomaly_caught = (anomaly_scores >= t).sum()
    total_flagged = normal_flagged + anomaly_caught
    precision = anomaly_caught / total_flagged if total_flagged > 0 else 0
    print(f"  {t:>10.3f}  {normal_flagged:>10d} ({100*normal_flagged/len(normal_scores):>5.2f}%)  "
          f"{anomaly_caught:>10d} ({100*anomaly_caught/len(anomaly_scores):>5.2f}%)  "
          f"{precision:>9.3f}")

# ── Daily volume estimate ───────────────────────────────────
# Val set covers days 127-153 = 27 days
val_days = 27
events_per_day = len(val_df) / val_days
print(f"\n  Events per day (avg): {events_per_day:.0f}")
print(f"\n  {'Threshold':>10s}  {'Alerts/day':>12s}")
print(f"  {'-'*25}")
for t in candidates:
    flagged = ((val_df["fused"] >= t).sum())
    per_day = flagged / val_days
    print(f"  {t:>10.3f}  {per_day:>11.1f}")