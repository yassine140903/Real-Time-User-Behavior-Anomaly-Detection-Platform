# tests/test_fusion.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score
from src.training.data_prep import prepare
from src.training.autoencoder import Autoencoder
from src.training.lstm import LSTMPredictor, build_sequences
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ================================================================
# STEP 1: Load everything
# ================================================================
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

# ================================================================
# STEP 2: AE scores on ALL test events
# ================================================================
with open(PROJECT_ROOT / "models" / "feature_cols.json") as f:
    _ = json.load(f)

ae = Autoencoder(len(feature_cols), hidden_dims=[20, 10])
ae.load_state_dict(torch.load(PROJECT_ROOT / "models" / "autoencoder.pt",
                               weights_only=True))
ae.eval()

X_test_t = torch.FloatTensor(data["X_test"])
with torch.no_grad():
    recon = ae(X_test_t)
    ae_raw = (recon - X_test_t).pow(2).mean(dim=1).numpy()

ae_pct = np.searchsorted(ae_ref, ae_raw, side="right") / len(ae_ref)

print(f"AE: scored {len(ae_pct)} test events")
print(f"  AE-only AUC: {roc_auc_score(data['y_test'], ae_pct):.4f}")

# ================================================================
# STEP 3: LSTM scores on test events that have sequences
# ================================================================
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

print(f"\nLSTM: scored {len(lstm_pct)} test sequences "
      f"(out of {len(ae_pct)} total test events)")

# ================================================================
# STEP 4: Fuse scores
# ================================================================
test_df = data["test_df"].copy().reset_index(drop=True)
test_df["ae_pct"] = ae_pct

test_meta["lstm_pct"] = lstm_pct

test_df = test_df.merge(
    test_meta[["event_id", "lstm_pct"]],
    on="event_id",
    how="left"
)

assert len(test_df) == len(ae_pct), \
    f"Merge changed row count: {len(test_df)} vs {len(ae_pct)}"

test_df["days_since_opening"] = test_df["client_id"].map(client_maturity)

def sigmoid_weight(days, midpoint=90, steepness=15):
    z = (days - midpoint) / steepness
    return 1.0 / (1.0 + np.exp(-z))

test_df["w_lstm"] = sigmoid_weight(test_df["days_since_opening"])

has_lstm = test_df["lstm_pct"].notna()
test_df["fused"] = test_df["ae_pct"]
test_df.loc[has_lstm, "fused"] = (
    (1 - test_df.loc[has_lstm, "w_lstm"]) * test_df.loc[has_lstm, "ae_pct"] +
    test_df.loc[has_lstm, "w_lstm"] * test_df.loc[has_lstm, "lstm_pct"]
)

# ================================================================
# STEP 5: Evaluate
# ================================================================
y_test = test_df["is_anomaly"].values.astype(int)

ae_only_auc = roc_auc_score(y_test, test_df["ae_pct"])
fused_auc = roc_auc_score(y_test, test_df["fused"])

lstm_subset = test_df[has_lstm]
lstm_only_auc = roc_auc_score(
    lstm_subset["is_anomaly"].values.astype(int),
    lstm_subset["lstm_pct"]
)

print(f"\n{'='*50}")
print(f"  AE-only AUC:     {ae_only_auc:.4f}  (all {len(test_df)} events)")
print(f"  LSTM-only AUC:   {lstm_only_auc:.4f}  ({has_lstm.sum()} events with sequences)")
print(f"  FUSED AUC:       {fused_auc:.4f}  (all {len(test_df)} events)")
print(f"{'='*50}")

print(f"\nLSTM coverage: {has_lstm.sum()}/{len(test_df)} "
      f"({100*has_lstm.mean():.1f}%) test events have sequences")
print(f"AE-only events: {(~has_lstm).sum()} (cold-start / sparse clients)")

print(f"\nSigmoid weight distribution:")
print(f"  w_lstm < 0.1 (AE-dominated):   {(test_df['w_lstm'] < 0.1).sum()}")
print(f"  0.1 <= w_lstm <= 0.9 (mixed):  "
      f"{((test_df['w_lstm'] >= 0.1) & (test_df['w_lstm'] <= 0.9)).sum()}")
print(f"  w_lstm > 0.9 (LSTM-dominated): {(test_df['w_lstm'] > 0.9).sum()}")