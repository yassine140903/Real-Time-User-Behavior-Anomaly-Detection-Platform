"""
Full pipeline: generate transactions + client metadata -> enrich -> stats.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.generator.generator import Generator
from src.enrichment import enrich
import pandas as pd

CONFIG = "config/archetype.yaml"
TX_PATH = "data/transactions.csv"
CLIENTS_PATH = "data/clients.csv"
ENRICHED_PATH = "data/enriched_events.csv"

# ── Step 1: Generate ──────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Generating synthetic transactions")
print("=" * 60)
t0 = time.time()

gen = Generator(
    config_path=CONFIG,
    num_clients=10000,
    num_branches=150,
    anomaly_rate=0.0013,
    sim_days=180,
)

events = gen.generate(start_date="2025-01-01")
gen.export_csv(events, output_path=TX_PATH)
gen.export_clients(output_path=CLIENTS_PATH)

t1 = time.time()
print(f"Generation time: {t1 - t0:.1f}s")

# ── Step 2: Enrich ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Enriching events (contrast features)")
print("=" * 60)

df = enrich(TX_PATH, CLIENTS_PATH, ENRICHED_PATH)

t2 = time.time()
print(f"Enrichment time: {t2 - t1:.1f}s")

# ── Step 3: Stats ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Dataset statistics")
print("=" * 60)

print(f"\nTotal events: {len(df)}")
print(f"Unique clients: {df['client_id'].nunique()}")

print(f"\nAnomaly events: {int(df['is_anomaly'].sum())}")
print(f"Anomaly rate: {df['is_anomaly'].mean()*100:.3f}%")

print("\nAnomaly types:")
anom = df[df["is_anomaly"] == True]
print(anom["anomaly_type"].value_counts().to_string())

print("\nDifficulty distribution:")
print(anom["difficulty"].value_counts().to_string())

# Feature-level sanity: do anomalous events actually differ?
feature_cols = [c for c in df.columns if c not in
                ("event_id", "client_id", "timestamp", "is_anomaly",
                 "anomaly_type", "difficulty")]

print(f"\nFeature columns ({len(feature_cols)}):")
print(feature_cols)

print("\n--- Feature means: Normal vs Anomaly ---")
normal = df[df["is_anomaly"] == False]
for col in ["z_amount", "amount_to_mean_ratio", "is_above_threshold",
            "is_near_threshold", "is_round_amount", "is_outside_hours",
            "tx_count_24h", "cumulative_amount_24h", "has_duplicate_recent",
            "near_threshold_count_7d", "is_new_counterparty",
            "same_employee_client_count_24h", "is_near_cheque_ceiling",
            "op_type_probability"]:
    if col in df.columns:
        n_mean = normal[col].mean()
        a_mean = anom[col].mean()
        print(f"  {col:40s}  normal={n_mean:8.3f}  anomaly={a_mean:8.3f}  ratio={a_mean/(n_mean+1e-9):6.2f}")

print(f"\nTotal pipeline time: {time.time() - t0:.1f}s")