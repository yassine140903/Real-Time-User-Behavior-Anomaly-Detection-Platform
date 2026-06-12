import pandas as pd

df = pd.read_csv("data/transactions.csv")

print("Total events:", len(df))
print("\nUnique clients:", df["client_id"].nunique())
print("\nOperation types:\n", df["operation_type"].value_counts())
print("\nAnomalies:", df["is_anomaly"].sum())
print("Anomaly rate:", round(df["is_anomaly"].mean() * 100, 3), "%")

print("\nAnomaly difficulty distribution:\n", df[df["is_anomaly"]==True]["anomaly_type"].value_counts())
print("\nAmount stats by operation:")
print(df.groupby("operation_type")["amount"].describe().round(2))