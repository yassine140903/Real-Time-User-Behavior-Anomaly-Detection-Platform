# scripts/compute_reference_scores.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import json
from pathlib import Path
from src.training.data_prep import prepare
from src.training.autoencoder import Autoencoder
from src.training.lstm import LSTMPredictor, build_sequences
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Load data ───────────────────────────────────────────────
data = prepare(str(PROJECT_ROOT / "data" / "enriched_events.csv"))
feature_cols = data["feature_cols"]
scaler = data["scaler"]

# ── AE reference scores ────────────────────────────────────
input_dim = len(feature_cols)
ae = Autoencoder(input_dim, hidden_dims=[20, 10])
ae.load_state_dict(torch.load(PROJECT_ROOT / "models" / "autoencoder.pt",
                               weights_only=True))
ae.eval()

X_train_t = torch.FloatTensor(data["X_train"])
with torch.no_grad():
    recon = ae(X_train_t)
    ae_scores = (recon - X_train_t).pow(2).mean(dim=1).numpy()

ae_sorted = np.sort(ae_scores)
print(f"AE reference: {len(ae_sorted)} scores")
print(f"  p50={np.median(ae_scores):.6f}  "
      f"p95={np.percentile(ae_scores, 95):.6f}  "
      f"p99={np.percentile(ae_scores, 99):.6f}")

# ── LSTM reference scores ──────────────────────────────────
with open(PROJECT_ROOT / "models" / "lstm_config.json") as f:
    cfg = json.load(f)

lstm = LSTMPredictor(cfg["input_dim"], cfg["hidden_dim"], cfg["num_layers"])
lstm.load_state_dict(torch.load(PROJECT_ROOT / "models" / "lstm.pt",
                                 weights_only=True))
lstm.eval()

# Rebuild sequences the same way training does
full_df = pd.concat([data["train_df"], data["val_df"], data["test_df"]]) \
            .sort_values("timestamp").reset_index(drop=True)
full_df[feature_cols] = scaler.transform(full_df[feature_cols].values)

sequences, targets, meta = build_sequences(full_df, feature_cols, cfg["seq_len"])

# Normal training sequences only
train_normal_mask = ((meta["day_offset"] <= 126) &
                     (meta["is_anomaly"] == False)).values

X_seq = torch.FloatTensor(sequences[train_normal_mask])
Y_tgt = torch.FloatTensor(targets[train_normal_mask])

# Score in batches to avoid OOM
lstm_scores = []
batch_size = 512
for i in range(0, len(X_seq), batch_size):
    with torch.no_grad():
        pred = lstm(X_seq[i:i+batch_size])
        batch_scores = (pred - Y_tgt[i:i+batch_size]).pow(2).mean(dim=1).numpy()
        lstm_scores.append(batch_scores)

lstm_scores = np.concatenate(lstm_scores)
lstm_sorted = np.sort(lstm_scores)

print(f"\nLSTM reference: {len(lstm_sorted)} scores")
print(f"  p50={np.median(lstm_scores):.6f}  "
      f"p95={np.percentile(lstm_scores, 95):.6f}  "
      f"p99={np.percentile(lstm_scores, 99):.6f}")

# ── Save ────────────────────────────────────────────────────
np.save(PROJECT_ROOT / "models" / "ae_ref_scores.npy", ae_sorted)
np.save(PROJECT_ROOT / "models" / "lstm_ref_scores.npy", lstm_sorted)
print(f"\nSaved to models/ae_ref_scores.npy and models/lstm_ref_scores.npy")