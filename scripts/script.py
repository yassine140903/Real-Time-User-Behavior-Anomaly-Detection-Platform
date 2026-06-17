import pandas as pd

# --- Transactions (existing check) ---
tx = pd.read_csv("data/transactions.csv")
print("=== TRANSACTIONS ===")
print(f"Total events: {len(tx)}")
print(f"Unique clients: {tx['client_id'].nunique()}")
print(f"Anomalies: {tx['is_anomaly'].sum()} ({tx['is_anomaly'].mean()*100:.3f}%)")
print(f"Operation types:\n{tx['operation_type'].value_counts()}\n")

# --- Clients ---
cl = pd.read_csv("data/clients_master.csv")
print("=== CLIENTS ===")
print(f"Total: {len(cl)}")
print(f"Archetypes:\n{cl['archetype'].value_counts()}")
print(f"Client types:\n{cl['client_type'].value_counts()}")
print(f"Opening date range: {cl['account_opening_date'].min()} to {cl['account_opening_date'].max()}\n")

# --- Accounts ---
ac = pd.read_csv("data/accounts.csv")
print("=== ACCOUNTS ===")
print(f"Total: {len(ac)}")
print(f"Account types:\n{ac['account_type'].value_counts()}")
print(f"All active? {(ac['status'] == 'active').all()}\n")

# --- Employees ---
em = pd.read_csv("data/employees_master.csv")
print("=== EMPLOYEES ===")
print(f"Total: {len(em)}")
print(f"Unique branches: {em['branch_id'].nunique()}")
print(f"Employees per branch: {em.groupby('branch_id').size().describe()}\n")

# --- Cross-check: are employee-branch pairs in transactions consistent? ---
tx_emp_branch = tx[['employee_id', 'branch_id']].drop_duplicates()
merged = tx_emp_branch.merge(em, on='employee_id', suffixes=('_tx', '_master'))
mismatches = merged[merged['branch_id_tx'] != merged['branch_id_master']]
print(f"=== CROSS-CHECK ===")
print(f"Employee-branch mismatches: {len(mismatches)} / {len(merged)}")