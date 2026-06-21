import numpy as np
import uuid
import datetime
import json
import csv
import calendar
from .archetype import load_archetypes
from .client import Client


class Generator:
    def __init__(self, config_path="config/archetype.yaml",
                 num_clients=1000, num_branches=150,
                 anomaly_rate=0.0013, sim_days=180):

        self.sim_days = sim_days
        self.anomaly_rate = anomaly_rate

        self.archetypes, self.proportions = load_archetypes(config_path)
        self.branch_pool = [f"BR-{str(i).zfill(3)}" for i in range(1, num_branches + 1)]
        self.employee_pool = [f"EMP-{str(i).zfill(4)}" for i in range(1, num_branches * 2 + 1)]

        self.clients = self._create_population(num_clients)
        self.client_lookup = {c.client_id: c for c in self.clients}
        self.anomaly_plan = self._plan_anomalies()

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------
    def _create_population(self, num_clients):
        clients = []
        for arch_name, proportion in self.proportions.items():
            count = int(num_clients * proportion)
            archetype = self.archetypes[arch_name]
            for _ in range(count):
                clients.append(Client(archetype, self.branch_pool))
        return clients

    # ------------------------------------------------------------------
    # Anomaly planning
    # ------------------------------------------------------------------
    def _plan_anomalies(self):
        plan = []

        total_events = sum(
            sum(c.frequency.values()) * self.sim_days / 30
            for c in self.clients
        )
        num_anomalous = int(total_events * self.anomaly_rate)

        scenarios = [
            # Easy (10%) — single-event, single-dimension
            {"name": "amount_spike",       "difficulty": "easy",   "duration": 1},
            {"name": "round_amount",       "difficulty": "easy",   "duration": 1},
            {"name": "duplicate_virement", "difficulty": "easy",   "duration": 1},
            # Medium (30%) — short-term multi-event
            {"name": "timing_anomaly",         "difficulty": "medium", "duration": 3},
            {"name": "frequency_burst",        "difficulty": "medium", "duration": 3},
            {"name": "cumulative_threshold",   "difficulty": "medium", "duration": 1},
            # Hard (60%) — sustained coordinated patterns
            {"name": "smurfing",                  "difficulty": "hard", "duration": 7},
            {"name": "cheque_structuring",        "difficulty": "hard", "duration": 7},
            {"name": "repeated_client_employee",  "difficulty": "hard", "duration": 1},
        ]

        difficulty_share = {"easy": 0.10, "medium": 0.30, "hard": 0.60}

        # Rough events-per-entry for budget tracking
        def _est_events(s):
            if s["name"] == "frequency_burst":
                return 15  # ~5x normal over 3 days
            elif s["name"] in ("smurfing", "cheque_structuring"):
                return s["duration"] * 2
            elif s["name"] == "cumulative_threshold":
                return 4
            elif s["name"] == "repeated_client_employee":
                return 6
            elif s["name"] == "duplicate_virement":
                return 2
            return 1

        for difficulty, share in difficulty_share.items():
            target = int(num_anomalous * share)
            pool = [s for s in scenarios if s["difficulty"] == difficulty]
            generated = 0

            while generated < target:
                scenario = pool[np.random.randint(len(pool))]
                client = self.clients[np.random.randint(len(self.clients))]
                start_day = np.random.randint(21, self.sim_days - 14)

                plan.append({
                    "client_id": client.client_id,
                    "scenario": scenario["name"],
                    "difficulty": difficulty,
                    "start_day": start_day,
                    "end_day": start_day + scenario["duration"] - 1,
                })
                generated += _est_events(scenario)

        return plan

    # ------------------------------------------------------------------
    # Main generation loop
    # ------------------------------------------------------------------
    def generate(self, start_date="2025-01-01"):
        events = []
        start = datetime.date.fromisoformat(start_date)

        # Build lookup: client_id -> list of plan entries
        anomaly_lookup = {}
        for entry in self.anomaly_plan:
            anomaly_lookup.setdefault(entry["client_id"], []).append(entry)

        for day_offset in range(self.sim_days):
            current_date = start + datetime.timedelta(days=day_offset)
            dom = current_date.day

            for client in self.clients:
                # Check anomaly window FIRST
                active = None
                if client.client_id in anomaly_lookup:
                    for p in anomaly_lookup[client.client_id]:
                        if p["start_day"] <= day_offset <= p["end_day"]:
                            active = p
                            break

                if active:
                    normal = self._generate_normal_day(client, current_date)
                    events.extend(self._apply_anomaly(client, current_date, active, normal))
                else:
                    dd = abs(dom - min(client.preferred_day,
                        calendar.monthrange(current_date.year, current_date.month)[1]))
                    if np.random.random() > self._daily_prob(client, dd):
                        continue
                    events.extend(self._generate_normal_day(client, current_date))

            if day_offset % 30 == 0:
                print(f"  Day {day_offset}/{self.sim_days} — {len(events)} events so far")

        return events

    # ------------------------------------------------------------------
    # Normal-day helpers
    # ------------------------------------------------------------------
    def _daily_prob(self, client, day_distance):
        base = sum(client.frequency.values()) / 30
        boost = np.exp(-0.5 * (day_distance / 3) ** 2)
        return min(base * boost, 0.95)

    def _generate_normal_day(self, client, date):
        dom = date.day
        dd = abs(dom - min(client.preferred_day,
            calendar.monthrange(date.year, date.month)[1]))
        if np.random.random() > self._daily_prob(client, dd):
            return []
        n_ops = max(1, np.random.poisson(sum(client.frequency.values()) / 30))
        return [self._normal_event(client, date) for _ in range(n_ops)]

    def _normal_event(self, client, date):
        ops = list(client.operation_mix.keys())
        probs = list(client.operation_mix.values())
        op = np.random.choice(ops, p=probs)

        amount = round(max(5, np.random.normal(
            client.amount_mean[op], client.amount_std[op])), 2)

        branch = (client.home_branch if np.random.random() < client.branch_loyalty
                  else np.random.choice(self.branch_pool))
        employee = np.random.choice(self.employee_pool)

        beneficiary = None
        if op in ("virement", "cheque"):
            if np.random.random() < client.counterparty_known_ratio:
                beneficiary = f"KNOWN-{np.random.randint(1, 6)}"
            else:
                beneficiary = f"NEW-{uuid.uuid4().hex[:8]}"

        hour = np.random.randint(8, 16)
        minute = np.random.randint(0, 60)
        ts = datetime.datetime.combine(date, datetime.time(hour, minute))

        return {
            "event_id": str(uuid.uuid4()),
            "client_id": client.client_id,
            "account_id": client.account_id,
            "employee_id": employee,
            "branch_id": branch,
            "timestamp": ts.isoformat(),
            "amount": amount,
            "currency": "TND",
            "channel": "guichet",
            "operation_type": op,
            "payload": json.dumps(self._payload(op, beneficiary)),
            "is_anomaly": False,
            "anomaly_type": None,
            "difficulty": None,
        }

    def _payload(self, op, beneficiary=None):
        if op == "retrait":
            return {"mode": "especes"}
        if op == "versement":
            return {"depositor_id": None}
        if op == "virement":
            return {"beneficiary_id": beneficiary, "motif": "normal"}
        if op == "cheque":
            return {"emitter_id": beneficiary,
                    "cheque_number": f"CHQ-{uuid.uuid4().hex[:8]}"}
        return {}

    # ------------------------------------------------------------------
    # Anomaly dispatch
    # ------------------------------------------------------------------
    def _mark(self, event, name, difficulty):
        event["is_anomaly"] = True
        event["anomaly_type"] = name
        event["difficulty"] = difficulty
        return event

    def _apply_anomaly(self, client, date, plan, normal_events):
        name = plan["scenario"]
        diff = plan["difficulty"]
        dispatch = {
            "amount_spike":              self._anom_amount_spike,
            "round_amount":              self._anom_round_amount,
            "duplicate_virement":        self._anom_duplicate_virement,
            "timing_anomaly":            self._anom_timing,
            "frequency_burst":           self._anom_frequency_burst,
            "cumulative_threshold":      self._anom_cumulative_threshold,
            "smurfing":                  self._anom_smurfing,
            "cheque_structuring":        self._anom_cheque_structuring,
            "repeated_client_employee":  self._anom_repeated_employee,
        }
        return dispatch[name](client, date, diff, normal_events)

    # === EASY ============================================================

    def _anom_amount_spike(self, client, date, diff, normal):
        if not normal:
            normal = [self._normal_event(client, date)]
        idx = np.random.randint(len(normal))
        normal[idx]["amount"] = round(normal[idx]["amount"] * np.random.uniform(5, 10), 2)
        self._mark(normal[idx], "amount_spike", diff)
        return normal

    def _anom_round_amount(self, client, date, diff, normal):
        if not normal:
            normal = [self._normal_event(client, date)]
        idx = np.random.randint(len(normal))
        normal[idx]["amount"] = float(np.random.choice([5000, 10000, 15000, 20000]))
        self._mark(normal[idx], "round_amount", diff)
        return normal

    def _anom_duplicate_virement(self, client, date, diff, normal):
        # Find or create a virement
        vir = None
        for e in normal:
            if e["operation_type"] == "virement":
                vir = e
                break
        if vir is None:
            vir = self._normal_event(client, date)
            vir["operation_type"] = "virement"
            ben = f"KNOWN-{np.random.randint(1, 6)}"
            vir["payload"] = json.dumps({"beneficiary_id": ben, "motif": "normal"})
            normal.append(vir)

        dup = vir.copy()
        dup["event_id"] = str(uuid.uuid4())
        ts = datetime.datetime.fromisoformat(vir["timestamp"])
        ts += datetime.timedelta(minutes=int(np.random.randint(5, 30)))
        dup["timestamp"] = ts.isoformat()

        self._mark(vir, "duplicate_virement", diff)
        self._mark(dup, "duplicate_virement", diff)
        normal.append(dup)
        return normal

    # === MEDIUM ==========================================================

    def _anom_timing(self, client, date, diff, normal):
        unusual = [5, 6, 7, 19, 20, 21]
        if not normal:
            normal = [self._normal_event(client, date)]
        for e in normal:
            h = int(np.random.choice(unusual))
            ts = datetime.datetime.fromisoformat(e["timestamp"]).replace(hour=h)
            e["timestamp"] = ts.isoformat()
            self._mark(e, "timing_anomaly", diff)
        return normal

    def _anom_frequency_burst(self, client, date, diff, normal):
        daily = max(1, sum(client.frequency.values()) / 30)
        burst = int(daily * np.random.uniform(5, 8))
        extras = []
        for _ in range(burst):
            e = self._normal_event(client, date)
            self._mark(e, "frequency_burst", diff)
            extras.append(e)
        return normal + extras

    def _anom_cumulative_threshold(self, client, date, diff, normal):
        target_total = np.random.uniform(10500, 15000)
        n_parts = np.random.randint(3, 6)
        splits = np.random.dirichlet(np.ones(n_parts)) * target_total
        extras = []
        for amt in splits:
            e = self._normal_event(client, date)
            e["operation_type"] = "versement"
            e["amount"] = round(min(float(amt), 9500), 2)
            e["payload"] = json.dumps({"depositor_id": None})
            self._mark(e, "cumulative_threshold", diff)
            extras.append(e)
        return normal + extras

    # === HARD ============================================================

    def _anom_smurfing(self, client, date, diff, normal):
        n = np.random.randint(1, 3)
        extras = []
        for _ in range(n):
            e = self._normal_event(client, date)
            e["operation_type"] = "versement"
            e["amount"] = round(np.random.uniform(8000, 9800), 2)
            e["payload"] = json.dumps({"depositor_id": None})
            self._mark(e, "smurfing", diff)
            extras.append(e)
        return normal + extras

    def _anom_cheque_structuring(self, client, date, diff, normal):
        n = np.random.randint(1, 3)
        extras = []
        for _ in range(n):
            e = self._normal_event(client, date)
            e["operation_type"] = "cheque"
            e["amount"] = round(np.random.uniform(25000, 29900), 2)
            emitter = f"NEW-{uuid.uuid4().hex[:8]}"
            e["payload"] = json.dumps({
                "emitter_id": emitter,
                "cheque_number": f"CHQ-{uuid.uuid4().hex[:8]}"
            })
            self._mark(e, "cheque_structuring", diff)
            extras.append(e)
        return normal + extras

    def _anom_repeated_employee(self, client, date, diff, normal):
        forced_emp = np.random.choice(self.employee_pool)
        n = np.random.randint(5, 9)
        extras = []
        for _ in range(n):
            e = self._normal_event(client, date)
            e["employee_id"] = forced_emp
            self._mark(e, "repeated_client_employee", diff)
            extras.append(e)
        return normal + extras

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_csv(self, events, output_path="data/transactions.csv"):
        columns = [
            "event_id", "client_id", "account_id", "employee_id",
            "branch_id", "timestamp", "amount", "currency", "channel",
            "operation_type", "payload", "is_anomaly", "anomaly_type", "difficulty"
        ]
        events.sort(key=lambda e: e["timestamp"])
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(events)
        print(f"Exported {len(events)} events to {output_path}")

    def export_clients(self, output_path="data/clients.csv"):
        """Export client personality as the oracle profile baseline."""
        columns = (
            ["client_id", "account_id", "archetype", "home_branch",
             "preferred_day", "branch_loyalty", "counterparty_known_ratio"]
            + [f"op_mix_{op}" for op in ["retrait", "versement", "virement", "cheque"]]
            + [f"amount_mean_{op}" for op in ["retrait", "versement", "virement", "cheque"]]
            + [f"amount_std_{op}" for op in ["retrait", "versement", "virement", "cheque"]]
            + [f"frequency_{op}" for op in ["retrait", "versement", "virement", "cheque"]]
        )
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for c in self.clients:
                row = {
                    "client_id": c.client_id,
                    "account_id": c.account_id,
                    "archetype": c.archetype.name,
                    "home_branch": c.home_branch,
                    "preferred_day": c.preferred_day,
                    "branch_loyalty": c.branch_loyalty,
                    "counterparty_known_ratio": c.counterparty_known_ratio,
                }
                for op in ["retrait", "versement", "virement", "cheque"]:
                    row[f"op_mix_{op}"] = round(c.operation_mix.get(op, 0), 6)
                    row[f"amount_mean_{op}"] = c.amount_mean.get(op, 0)
                    row[f"amount_std_{op}"] = c.amount_std.get(op, 0)
                    row[f"frequency_{op}"] = c.frequency.get(op, 0)
                writer.writerow(row)
        print(f"Exported {len(self.clients)} clients to {output_path}")