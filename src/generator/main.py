from pathlib import Path
from .generator import Generator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "archetype.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data" / "transactions.csv"

gen = Generator(
    config_path=str(CONFIG_PATH),
    num_clients=10000,
    sim_days=180,
    anomaly_rate=0.0013
)

events = gen.generate(start_date="2025-01-01")
gen.export_csv(events, output_path=str(OUTPUT_PATH))
gen.export_clients(output_path=str(PROJECT_ROOT / "data" / "clients_master.csv"))
gen.export_accounts(output_path=str(PROJECT_ROOT / "data" / "accounts.csv"))
gen.export_employees(output_path=str(PROJECT_ROOT / "data" / "employees_master.csv"))