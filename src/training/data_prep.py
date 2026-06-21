"""
Data preparation for model training.

Pipeline:
  1. Load enriched events CSV
  2. Warm-up filter (drop first N days — buffer features need time to populate)
  3. Identify feature columns (exclude identifiers + labels)
  4. Chronological split (train / val / test by day offset)
  5. StandardScaler fit on normal training data only
  6. Return numpy arrays + metadata for training and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


EXCLUDE_COLS = {"event_id", "client_id", "timestamp",
                "is_anomaly", "anomaly_type", "difficulty"}


def prepare(enriched_path: str,
            sim_start: str = "2025-01-01",
            warmup_days: int = 21,
            train_end_day: int = 126,
            val_end_day: int = 153):
    """
    Load, filter, split, and scale the enriched dataset.

    Returns a dict with:
        X_train, X_val, X_test     — scaled numpy arrays (train = normal only)
        y_val, y_test              — binary labels (0=normal, 1=anomaly)
        feature_cols               — list of feature column names
        scaler                     — fitted StandardScaler
        train_df, val_df, test_df  — original DataFrames (for per-scenario eval)
    """

    print("Loading enriched data...")
    df = pd.read_csv(enriched_path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    sim_start_ts = pd.Timestamp(sim_start)
    df["day_offset"] = (df["timestamp"] - sim_start_ts).dt.days

    # Warm-up filter
    df = df[df["day_offset"] >= warmup_days].reset_index(drop=True)
    print(f"After warm-up filter (>= day {warmup_days}): {len(df)} events")

    # Feature columns
    feature_cols = sorted([c for c in df.columns
                           if c not in EXCLUDE_COLS and c != "day_offset"])
    print(f"Feature columns ({len(feature_cols)})")

    # Chronological split
    train_df = df[df["day_offset"] <= train_end_day].copy()
    val_df   = df[(df["day_offset"] > train_end_day) &
                  (df["day_offset"] <= val_end_day)].copy()
    test_df  = df[df["day_offset"] > val_end_day].copy()

    # Normal-only training set
    train_normal = train_df[train_df["is_anomaly"] == False]

    print(f"\nTrain:  {len(train_df):>7} events  |  {int(train_df['is_anomaly'].sum()):>4} anomalies")
    print(f"  (normal only for training: {len(train_normal)})")
    print(f"Val:    {len(val_df):>7} events  |  {int(val_df['is_anomaly'].sum()):>4} anomalies")
    print(f"Test:   {len(test_df):>7} events  |  {int(test_df['is_anomaly'].sum()):>4} anomalies")

    # Scale: fit on normal training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_normal[feature_cols].values)
    X_val   = scaler.transform(val_df[feature_cols].values)
    X_test  = scaler.transform(test_df[feature_cols].values)

    y_val  = val_df["is_anomaly"].values.astype(int)
    y_test = test_df["is_anomaly"].values.astype(int)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_val": y_val,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }