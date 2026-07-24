"""
Unified training entry point with MLflow tracking and promotion gate.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mlflow
import torch

from src.training.data_prep import prepare
from src.training.autoencoder import train as train_ae, evaluate as evaluate_ae, compute_scores as compute_ae_scores
from src.training.lstm import train as train_lstm, evaluate as evaluate_lstm, compute_scores as compute_lstm_scores

PRODUCTION_METRICS_PATH = "models/production_metrics.json"
mlflow.set_experiment("amen-anomaly-detection")

def load_production_metrics():
    if os.path.exists(PRODUCTION_METRICS_PATH):
        with open(PRODUCTION_METRICS_PATH) as f:
            return json.load(f)
    return {"ae_auc": 0.0, "lstm_auc": 0.0}

def save_production_metrics(metrics):
    with open(PRODUCTION_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

# ── Data Prep ─────────────────────────────────────────────────────────
print("=" * 60)
print("DATA PREP")
print("=" * 60)

data = prepare("data/training_features.csv")
prod_metrics = load_production_metrics()

# ── Autoencoder ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("AUTOENCODER TRAINING")
print("=" * 60)

with mlflow.start_run(run_name="autoencoder") as ae_run:
    ae_params = {"hidden_dims": [20, 10], "epochs": 50, "batch_size": 256, "lr": 1e-3}
    mlflow.log_params(ae_params)
    mlflow.log_param("model_type", "autoencoder")
    mlflow.log_param("n_features", data["X_train"].shape[1])
    mlflow.log_param("n_train_samples", data["X_train"].shape[0])

    ae_model = train_ae(data, **ae_params)
    ae_test_auc, ae_scores = evaluate_ae(ae_model, data)

    mlflow.log_metric("test_auc", ae_test_auc)
    print(f"\nAutoencoder test AUC: {ae_test_auc:.4f}")

    # Log shared artifacts
    mlflow.log_artifact("models/scaler.pkl")
    mlflow.log_artifact("models/feature_cols.json")

    # Compute reference scores
    ae_ref_scores = np.sort(compute_ae_scores(ae_model, data["X_train"]))

    # Promotion gate
    prev_auc = prod_metrics["ae_auc"]
    if ae_test_auc >= prev_auc:
        torch.save(ae_model.state_dict(), "models/autoencoder.pt")
        np.save("models/ae_ref_scores.npy", ae_ref_scores)
        prod_metrics["ae_auc"] = ae_test_auc
        mlflow.log_metric("promoted", 1)
        mlflow.set_tag("status", "promoted")
        print(f"  PROMOTED: {ae_test_auc:.4f} >= {prev_auc:.4f}")
    else:
        mlflow.log_metric("promoted", 0)
        mlflow.set_tag("status", "rejected")
        print(f"  REJECTED: {ae_test_auc:.4f} < {prev_auc:.4f}")

    mlflow.log_metric("production_auc", prev_auc)

# ── LSTM ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LSTM TRAINING")
print("=" * 60)

with mlflow.start_run(run_name="lstm") as lstm_run:
    lstm_params = {"seq_len": 10, "hidden_dim": 64, "num_layers": 2, "epochs": 50, "batch_size": 256, "lr": 1e-3}
    mlflow.log_params(lstm_params)
    mlflow.log_param("model_type", "lstm")
    mlflow.log_param("n_features", data["X_train"].shape[1])
    mlflow.log_param("n_train_samples", data["X_train"].shape[0])

    lstm_model, lstm_test_data = train_lstm(data, **lstm_params)
    lstm_test_auc, lstm_scores = evaluate_lstm(lstm_model, lstm_test_data)

    mlflow.log_metric("test_auc", lstm_test_auc)
    print(f"\nLSTM test AUC: {lstm_test_auc:.4f}")

    mlflow.log_artifact("models/scaler.pkl")
    mlflow.log_artifact("models/feature_cols.json")

    lstm_ref_scores = np.sort(compute_lstm_scores(
        lstm_model, lstm_test_data["X_train_seq"], lstm_test_data["Y_train"]))

    prev_auc = prod_metrics["lstm_auc"]
    if lstm_test_auc >= prev_auc:
        torch.save(lstm_model.state_dict(), "models/lstm.pt")
        np.save("models/lstm_ref_scores.npy", lstm_ref_scores)
        prod_metrics["lstm_auc"] = lstm_test_auc
        mlflow.log_metric("promoted", 1)
        mlflow.set_tag("status", "promoted")
        print(f"  PROMOTED: {lstm_test_auc:.4f} >= {prev_auc:.4f}")
    else:
        mlflow.log_metric("promoted", 0)
        mlflow.set_tag("status", "rejected")
        print(f"  REJECTED: {lstm_test_auc:.4f} < {prev_auc:.4f}")

    mlflow.log_metric("production_auc", prev_auc)

# ── Save updated production metrics ──────────────────────────────────
save_production_metrics(prod_metrics)

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  {'Model':<15s}  {'Test AUC':>8s}  {'Prod AUC':>8s}  {'Status':>10s}")
print(f"  {'-'*45}")
for name, run_id in [("Autoencoder", ae_run), ("LSTM", lstm_run)]:
    run_data = mlflow.get_run(run_id.info.run_id)
    status = run_data.data.tags.get("status", "unknown")
    test_auc = run_data.data.metrics["test_auc"]
    prod_auc = run_data.data.metrics["production_auc"]
    print(f"  {name:<15s}  {test_auc:>8.4f}  {prod_auc:>8.4f}  {status:>10s}")