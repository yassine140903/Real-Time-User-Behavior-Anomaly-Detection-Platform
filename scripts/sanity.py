import pandas as pd

df = pd.read_csv("data/training_features.csv")

print("Shape:", df.shape)
print("\n--- Feature stats ---")
print(df.describe().round(4).to_string())

print("\n--- Zero-variance columns ---")
for col in df.select_dtypes(include="number").columns:
    if df[col].std() == 0:
        print(f"  {col}: constant at {df[col].iloc[0]}")

print("\n--- z_amount distribution ---")
print(df["z_amount"].describe())

print("\n--- op type one-hots sum ---")
print(df[["op_retrait","op_versement","op_virement","op_cheque"]].sum())