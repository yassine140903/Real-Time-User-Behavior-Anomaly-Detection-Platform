import pandas as pd
tx = pd.read_csv("data/transactions.csv")
pairs = tx.groupby("employee_id")["branch_id"].nunique()
print(f"Employees with >1 branch: {(pairs > 1).sum()}")
print(f"Total unique employees: {len(pairs)}")