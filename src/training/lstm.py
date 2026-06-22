"""
LSTM next-event predictor for anomaly detection.

Approach:
  - Build per-client sequences of length seq_len
  - Train LSTM to predict next event features from sequence context
  - Anomaly score = prediction error (MSE) on the target event
  - Trained on normal-only sequences; labels used for evaluation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import time


# ── Sequence Building ────────────────────────────────────────────────

def build_sequences(df, feature_cols, seq_len=10):
    """
    Build sliding-window sequences per client.

    Returns:
        sequences: np.array (N, seq_len, n_features)
        targets:   np.array (N, n_features)
        meta:      DataFrame with day_offset, is_anomaly, anomaly_type,
                   difficulty, client_id for each target event
    """
    sequences = []
    targets = []
    meta_rows = []

    for client_id, group in df.groupby("client_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)

        if len(group) <= seq_len:
            continue  # Not enough events for a full sequence

        feats = group[feature_cols].values  # (n_events, n_features)

        for i in range(seq_len, len(group)):
            seq = feats[i - seq_len: i]     # (seq_len, n_features)
            target = feats[i]               # (n_features,)

            row = group.iloc[i]
            sequences.append(seq)
            targets.append(target)
            meta_rows.append({
                "event_id": row["event_id"],      # ← add this
                "day_offset": row["day_offset"],
                "is_anomaly": row["is_anomaly"],
                "anomaly_type": row.get("anomaly_type"),
                "difficulty": row.get("difficulty"),
                "client_id": client_id,
            })

    return (np.array(sequences, dtype=np.float32),
            np.array(targets, dtype=np.float32),
            pd.DataFrame(meta_rows))


# ── Model ────────────────────────────────────────────────────────────

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        last = out[:, -1, :]           # last timestep hidden state
        return self.fc(last)           # (batch, input_dim)


# ── Training ─────────────────────────────────────────────────────────

def train(data, seq_len=10, hidden_dim=64, num_layers=2,
          epochs=50, batch_size=256, lr=1e-3):
    """
    Build sequences, split, train LSTM on normal sequences.

    Args:
        data: dict from data_prep.prepare() — needs df with day_offset
    """
    feature_cols = data["feature_cols"]
    scaler = data["scaler"]

    # We need the full dataframe with day_offset for sequence building
    # Rebuild from train/val/test with scaling applied
    train_df = data["train_df"].copy()
    val_df = data["val_df"].copy()
    test_df = data["test_df"].copy()

    full_df = pd.concat([train_df, val_df, test_df]).sort_values("timestamp").reset_index(drop=True)

    # Scale features in-place for sequence building
    full_df[feature_cols] = scaler.transform(full_df[feature_cols].values)

    print(f"Building sequences (seq_len={seq_len})...")
    t0 = time.time()
    sequences, targets, meta = build_sequences(full_df, feature_cols, seq_len)
    print(f"  Built {len(sequences)} sequences in {time.time()-t0:.1f}s")

    # Split by target event's day_offset
    train_end = 126
    val_end = 153

    train_mask = (meta["day_offset"] <= train_end) & (meta["is_anomaly"] == False)
    val_mask   = (meta["day_offset"] > train_end) & (meta["day_offset"] <= val_end)
    test_mask  = (meta["day_offset"] > val_end)

    X_train_seq = torch.FloatTensor(sequences[train_mask.values])
    Y_train     = torch.FloatTensor(targets[train_mask.values])
    X_val_seq   = torch.FloatTensor(sequences[val_mask.values])
    Y_val       = torch.FloatTensor(targets[val_mask.values])
    X_test_seq  = torch.FloatTensor(sequences[test_mask.values])
    Y_test      = torch.FloatTensor(targets[test_mask.values])

    y_val_labels  = meta[val_mask.values]["is_anomaly"].values.astype(int)
    y_test_labels = meta[test_mask.values]["is_anomaly"].values.astype(int)
    test_meta     = meta[test_mask.values].reset_index(drop=True)

    print(f"\n  Train sequences: {len(X_train_seq)} (normal only)")
    print(f"  Val sequences:   {len(X_val_seq)} ({y_val_labels.sum()} anomalous)")
    print(f"  Test sequences:  {len(X_test_seq)} ({y_test_labels.sum()} anomalous)")

    # Model
    input_dim = len(feature_cols)
    model = LSTMPredictor(input_dim, hidden_dim, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(X_train_seq, Y_train),
                        batch_size=batch_size, shuffle=True)

    print(f"\n  Architecture: LSTM({input_dim}, hidden={hidden_dim}, "
          f"layers={num_layers}) → Linear → {input_dim}")
    print(f"  Training {epochs} epochs\n")

    best_val_auc = 0
    best_state = None

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for seq_batch, target_batch in loader:
            pred = model(seq_batch)
            loss = criterion(pred, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_seq)
                val_scores = (val_pred - Y_val).pow(2).mean(dim=1).numpy()
                val_auc = (roc_auc_score(y_val_labels, val_scores)
                           if y_val_labels.sum() > 0 else 0.0)

            marker = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                marker = " ★"

            print(f"  Epoch {epoch+1:>3}/{epochs}  "
                  f"loss={avg_loss:.6f}  val_AUC={val_auc:.4f}{marker}")

    print(f"\nTraining time: {time.time()-t0:.1f}s")
    print(f"Best validation AUC: {best_val_auc:.4f}")

    # Load best and save
    model.load_state_dict(best_state)

    import os, joblib, json
    os.makedirs("models", exist_ok=True)
    torch.save(best_state, "models/lstm.pt")
    json.dump({"seq_len": seq_len, "hidden_dim": hidden_dim,
               "num_layers": num_layers, "input_dim": input_dim},
              open("models/lstm_config.json", "w"))
    print("Saved LSTM model + config to models/")

    return model, {
        "X_test_seq": X_test_seq,
        "Y_test": Y_test,
        "y_test_labels": y_test_labels,
        "test_meta": test_meta,
    }


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate(model, test_data):
    """Evaluate LSTM on test set with per-scenario breakdown."""

    X_test_seq    = test_data["X_test_seq"]
    Y_test        = test_data["Y_test"]
    y_test_labels = test_data["y_test_labels"]
    test_meta     = test_data["test_meta"]

    model.eval()
    with torch.no_grad():
        pred = model(X_test_seq)
        scores = (pred - Y_test).pow(2).mean(dim=1).numpy()

    test_auc = roc_auc_score(y_test_labels, scores)
    print(f"\n{'='*40}")
    print(f"  LSTM TEST AUC: {test_auc:.4f}")
    print(f"{'='*40}")

    # Per-scenario
    test_meta = test_meta.copy()
    test_meta["score"] = scores
    normal_mask = test_meta["is_anomaly"] == False

    print(f"\nPer-scenario AUC (test set):")
    print(f"  {'Scenario':<30s}  {'AUC':>6s}  {'Count':>5s}  {'Difficulty'}")
    print(f"  {'-'*65}")

    anom = test_meta[test_meta["is_anomaly"] == True]
    if len(anom) > 0:
        groups = anom.groupby(["anomaly_type", "difficulty"]).size().reset_index(name="count")
        for _, row in groups.iterrows():
            atype, diff, count = row["anomaly_type"], row["difficulty"], row["count"]
            anom_mask = test_meta["anomaly_type"] == atype
            subset = test_meta[anom_mask | normal_mask]
            y_sub = subset["is_anomaly"].values.astype(int)
            s_sub = subset["score"].values
            if 0 < y_sub.sum() < len(y_sub):
                auc = roc_auc_score(y_sub, s_sub)
            else:
                auc = float("nan")
            print(f"  {atype:<30s}  {auc:>6.4f}  {count:>5d}  {diff}")

    # Per-difficulty
    print(f"\nPer-difficulty AUC:")
    for diff in ["easy", "medium", "hard"]:
        diff_mask = test_meta["difficulty"] == diff
        if diff_mask.sum() == 0:
            continue
        subset = test_meta[diff_mask | normal_mask]
        y_sub = subset["is_anomaly"].values.astype(int)
        s_sub = subset["score"].values
        auc = (roc_auc_score(y_sub, s_sub)
               if 0 < y_sub.sum() < len(y_sub) else float("nan"))
        print(f"  {diff:<10s}  AUC={auc:.4f}  n={int(diff_mask.sum())}")

    # Score distribution
    print(f"\nScore distribution:")
    print(f"  Normal  — mean={scores[y_test_labels==0].mean():.6f}  "
          f"std={scores[y_test_labels==0].std():.6f}  "
          f"p95={np.percentile(scores[y_test_labels==0], 95):.6f}")
    if y_test_labels.sum() > 0:
        print(f"  Anomaly — mean={scores[y_test_labels==1].mean():.6f}  "
              f"std={scores[y_test_labels==1].std():.6f}  "
              f"p5={np.percentile(scores[y_test_labels==1], 5):.6f}")

    return test_auc, scores
