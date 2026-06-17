import pandas as pd
df = pd.read_csv("data/enriched_events.csv")
print("Shape:", df.shape)
print("\nNull counts (top 10):")
print(df.isnull().sum().sort_values(ascending=False).head(10))
print("\nAnomaly rate:", round(df["is_anomaly"].mean() * 100, 3), "%")
print("\nSample z_amount_30d stats:")
print(df["z_amount_30d"].describe())