import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Config ──────────────────────────────────────────────────
WARMUP_DAYS = 21
TRAIN_CUTOFF = 126
VAL_CUTOFF = 153
LSTM_SEQ_LEN = 50

META_COLS = [
    "enriched_event_id", "raw_event_id", "client_id", "employee_id",
    "branch_id", "raw_event_timestamp",
    "client_profile_snapshot_at", "employee_profile_snapshot_at",
    "is_anomaly", "anomaly_type",
]

DROP_COLS = []  # nothing extra to drop — enrichment already cleaned these

DROP_COLS = ["currency", "channel", "recent_events_buffer"]

# ── Step 1: Load & filter warm-up ───────────────────────────
def load_and_filter(path=None):
    if path is None:
        path = PROJECT_ROOT / "data" / "enriched_events.csv"
    
    df = pd.read_csv(path, parse_dates=["raw_event_timestamp"])
    start_date = df["raw_event_timestamp"].min().normalize()
    df["day_offset"] = (df["raw_event_timestamp"] - start_date).dt.days
    
    df = df[df["day_offset"] >= WARMUP_DAYS].copy()
    print(f"After warm-up filter: {len(df)} events (dropped first {WARMUP_DAYS} days)")
    return df

# ── Step 2: Preprocess ──────────────────────────────────────
def preprocess(df):
    # Drop constants and non-numeric structures
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    
    # One-hot encode operation_type
    df = pd.get_dummies(df, columns=["operation_type", "client_archetype"], prefix=["op", "arch"], dtype=float)

    return df

# ── Step 3: Temporal split ──────────────────────────────────
def temporal_split(df):
    train = df[df["day_offset"] <= TRAIN_CUTOFF].copy()
    val = df[(df["day_offset"] > TRAIN_CUTOFF) & (df["day_offset"] <= VAL_CUTOFF)].copy()
    test = df[df["day_offset"] > VAL_CUTOFF].copy()
    
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Train anomalies: {train['is_anomaly'].sum()} | "
          f"Val anomalies: {val['is_anomaly'].sum()} | "
          f"Test anomalies: {test['is_anomaly'].sum()}")
    return train, val, test

# ── Step 4: Feature columns & scaling ───────────────────────
def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS and c != "day_offset"]

def fit_scaler(train, feature_cols):
    scaler = StandardScaler()
    scaler.fit(train[feature_cols].values)
    return scaler

def scale(df, scaler, feature_cols):
    scaled = df.copy()
    scaled[feature_cols] = scaler.transform(df[feature_cols].values)
    return scaled

# ── Step 5: Autoencoder datasets ────────────────────────────
def prepare_autoencoder(train, val, test, feature_cols):
    ae_train = train[train["is_anomaly"] == False][feature_cols].values
    
    ae_val_X = val[feature_cols].values
    ae_val_y = val["is_anomaly"].values.astype(int)
    
    ae_test_X = test[feature_cols].values
    ae_test_y = test["is_anomaly"].values.astype(int)
    
    print(f"\nAutoencoder — Train: {ae_train.shape} (all normal)")
    print(f"Autoencoder — Val: {ae_val_X.shape[0]} events ({ae_val_y.sum()} anomalies)")
    print(f"Autoencoder — Test: {ae_test_X.shape[0]} events ({ae_test_y.sum()} anomalies)")
    
    return ae_train, ae_val_X, ae_val_y, ae_test_X, ae_test_y

def build_sequences(df, feature_cols, seq_len):
    sequences, targets, labels, day_offsets, window_clean = [], [], [], [], []
    
    df = df.sort_values(["client_id", "raw_event_timestamp"])
    
    for client_id, group in df.groupby("client_id"):
        events = group[feature_cols].values
        anomaly_flags = group["is_anomaly"].values
        offsets = group["day_offset"].values
        
        if len(events) < seq_len + 1:
            continue
        
        for i in range(len(events) - seq_len):
            sequences.append(events[i : i + seq_len])
            targets.append(events[i + seq_len])
            labels.append(anomaly_flags[i + seq_len])
            day_offsets.append(offsets[i + seq_len])
            window_clean.append(not anomaly_flags[i : i + seq_len + 1].any())
    
    return (np.array(sequences), np.array(targets),
            np.array(labels), np.array(day_offsets), np.array(window_clean))


def prepare_lstm(df, feature_cols):
    print("\nBuilding LSTM sequences...")
    
    seqs, tgts, lbls, offsets, clean = build_sequences(df, feature_cols, LSTM_SEQ_LEN)
    
    # Split by target day_offset
    train_mask = (offsets <= TRAIN_CUTOFF) & clean  # normal-only for training
    val_mask = (offsets > TRAIN_CUTOFF) & (offsets <= VAL_CUTOFF)
    test_mask = offsets > VAL_CUTOFF
    
    print(f"LSTM — Train: {train_mask.sum()} sequences (all normal)")
    print(f"LSTM — Val: {val_mask.sum()} sequences ({lbls[val_mask].sum()} anomalous targets)")
    print(f"LSTM — Test: {test_mask.sum()} sequences ({lbls[test_mask].sum()} anomalous targets)")
    
    return (seqs[train_mask], tgts[train_mask],
            seqs[val_mask], tgts[val_mask], lbls[val_mask],
            seqs[test_mask], tgts[test_mask], lbls[test_mask])


def save_datasets(ae_data, lstm_data, feature_cols, output_dir=None):
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "training"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ae_train, ae_val_X, ae_val_y, ae_test_X, ae_test_y = ae_data
    (lstm_train_X, lstm_train_y,
     lstm_val_X, lstm_val_y, lstm_val_labels,
     lstm_test_X, lstm_test_y, lstm_test_labels) = lstm_data
    
    np.savez(output_dir / "autoencoder.npz",
             train=ae_train,
             val_X=ae_val_X, val_y=ae_val_y,
             test_X=ae_test_X, test_y=ae_test_y)
    
    np.savez(output_dir / "lstm.npz",
             train_X=lstm_train_X, train_y=lstm_train_y,
             val_X=lstm_val_X, val_y=lstm_val_y, val_labels=lstm_val_labels,
             test_X=lstm_test_X, test_y=lstm_test_y, test_labels=lstm_test_labels)
    
    # Save feature column names for later reference
    pd.Series(feature_cols).to_csv(output_dir / "feature_cols.csv", index=False)
    
    print(f"\nSaved to {output_dir}")

# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_and_filter()
    df = preprocess(df)
    train, val, test = temporal_split(df)
    
    feature_cols = get_feature_cols(train)
    print(f"\nFeature columns: {len(feature_cols)}")
    
    scaler = fit_scaler(train[train["is_anomaly"] == False], feature_cols)
    train = scale(train, scaler, feature_cols)
    val = scale(val, scaler, feature_cols)
    test = scale(test, scaler, feature_cols)
    
    ae_data = prepare_autoencoder(train, val, test, feature_cols)
    
    # LSTM needs full scaled timeline
    full_scaled = pd.concat([train, val, test])
    lstm_data = prepare_lstm(full_scaled, feature_cols)

    save_datasets(ae_data, lstm_data, feature_cols)
    
    print(f"\nFeature dim: {len(feature_cols)}")
    print("Done. Ready for training.")