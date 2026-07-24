"""
Autoencoder for anomaly detection.

Trained on normal events only. Anomaly score = mean squared reconstruction error.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import time
import joblib
import json
import os


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[20, 10]):
        super().__init__()

        # Encoder
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*enc)

        # Decoder (mirror)
        dec = []
        for h in reversed(hidden_dims[:-1]):
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train(data, hidden_dims=[20, 10], epochs=50, batch_size=256, lr=1e-3):
    """
    Train autoencoder on normal data. Returns trained model + best state dict.

    Args:
        data: dict from data_prep.prepare()
    """
    X_train_t = torch.FloatTensor(data["X_train"])
    X_val_t   = torch.FloatTensor(data["X_val"])
    y_val     = data["y_val"]

    input_dim = data["X_train"].shape[1]
    model = Autoencoder(input_dim, hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(X_train_t),
                        batch_size=batch_size, shuffle=True)

    print(f"Architecture: {input_dim} → {' → '.join(map(str, hidden_dims))}"
          f" → {' → '.join(map(str, reversed(hidden_dims[:-1])))} → {input_dim}")
    print(f"Training on {len(X_train_t)} normal events, {epochs} epochs\n")

    best_val_auc = 0
    best_state = None

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for (batch,) in loader:
            recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_recon = model(X_val_t)
                val_scores = (val_recon - X_val_t).pow(2).mean(dim=1).numpy()
                val_auc = (roc_auc_score(y_val, val_scores)
                           if y_val.sum() > 0 else 0.0)

            marker = " ★" if val_auc > best_val_auc else ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            print(f"  Epoch {epoch+1:>3}/{epochs}  "
                  f"loss={avg_loss:.6f}  val_AUC={val_auc:.4f}{marker}")

    print(f"\nTraining time: {time.time() - t0:.1f}s")
    print(f"Best validation AUC: {best_val_auc:.4f}")

    # Load best
    model.load_state_dict(best_state)
    
    # Save artifacts
    os.makedirs("models", exist_ok=True)
    torch.save(best_state, "models/autoencoder.pt")
    joblib.dump(data["scaler"], "models/scaler.pkl")
    with open("models/feature_cols.json", "w") as f:
        json.dump(data["feature_cols"], f)
    print(f"Saved model + scaler + feature_cols to models/")
    
    return model


def compute_scores(model, X):
    """Reconstruction error (per-sample MSE) for feature array X."""
    X_t = X if isinstance(X, torch.Tensor) else torch.FloatTensor(X)
    model.eval()
    with torch.no_grad():
        recon = model(X_t)
        scores = (recon - X_t).pow(2).mean(dim=1).numpy()
    return scores


def evaluate(model, data):
    """Evaluate on test set. Print overall + per-scenario AUC."""

    y_test   = data["y_test"]
    test_df  = data["test_df"].copy()

    scores = compute_scores(model, data["X_test"])

    # Overall
    test_auc = roc_auc_score(y_test, scores)
    print(f"\n{'='*40}")
    print(f"  OVERALL TEST AUC: {test_auc:.4f}")
    print(f"{'='*40}")

    # Per-scenario
    test_df["score"] = scores
    normal_mask = test_df["is_anomaly"] == False

    print(f"\nPer-scenario AUC (test set):")
    print(f"  {'Scenario':<30s}  {'AUC':>6s}  {'Count':>5s}")
    print(f"  {'-'*65}")

    anom_groups = (test_df[test_df["is_anomaly"] == True]
                   .groupby("anomaly_type")
                   .size().reset_index(name="count"))

    for _, row in anom_groups.iterrows():
        atype, count = row["anomaly_type"], row["count"]
        anom_mask = test_df["anomaly_type"] == atype
        subset = test_df[anom_mask | normal_mask]
        y_sub = subset["is_anomaly"].values.astype(int)
        s_sub = subset["score"].values

        if y_sub.sum() > 0 and y_sub.sum() < len(y_sub):
            auc = roc_auc_score(y_sub, s_sub)
        else:
            auc = float("nan")
        print(f"  {atype:<30s}  {auc:>6.4f}  {count:>5d}")

    # Score distribution
    print(f"\nScore distribution:")
    print(f"  Normal  — mean={scores[y_test==0].mean():.6f}  "
          f"std={scores[y_test==0].std():.6f}  "
          f"p95={np.percentile(scores[y_test==0], 95):.6f}")
    if y_test.sum() > 0:
        print(f"  Anomaly — mean={scores[y_test==1].mean():.6f}  "
              f"std={scores[y_test==1].std():.6f}  "
              f"p5={np.percentile(scores[y_test==1], 5):.6f}")

    return test_auc, scores
