# add_account_opening_date.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

df = pd.read_csv("data/clients.csv")
sim_start = datetime(2025, 1, 1)

# Days before sim_start that account was opened, per archetype
# (min_days, max_days)
ranges = {
    "retiree":       (1825, 7300),   # 5-20 years
    "salaried":      (14, 3650),     # 2 weeks - 10 years
    "student":       (30, 1095),     # 1 month - 3 years
    "small_business": (14, 3650),    # 2 weeks - 10 years
    "big_business":  (730, 5475),    # 2-15 years
}

days_before = np.zeros(len(df), dtype=int)

for archetype, (lo, hi) in ranges.items():
    mask = df["archetype"] == archetype
    n = mask.sum()
    days_before[mask] = np.random.randint(lo, hi, size=n)

df["account_opening_date"] = [
    (sim_start - timedelta(days=int(d))).strftime("%Y-%m-%d")
    for d in days_before
]

# Quick check
df["days_since_opening"] = days_before
print(df.groupby("archetype")["days_since_opening"].describe().round(0))
print(f"\nClients under 90 days: {(days_before < 90).sum()}")
print(f"Clients under 180 days: {(days_before < 180).sum()}")

df.drop(columns=["days_since_opening"]).to_csv("data/clients.csv", index=False)
print("\nSaved updated clients.csv")