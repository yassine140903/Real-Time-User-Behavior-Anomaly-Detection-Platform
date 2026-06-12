import numpy as np
import uuid
from .archetype import Archetype, load_archetypes
from .client import Client
import datetime
import json
import calendar
import csv

class Generator:
    def __init__(self, config_path="config/archetypes.yaml", 
                 num_clients=1000, 
                 num_branches=150,
                 anomaly_rate=0.0013,
                 difficulty_distribution=None,
                 sim_days=180):
        
        self.sim_days = sim_days
        self.anomaly_rate = anomaly_rate
        self.difficulty_distribution = difficulty_distribution or {
            "easy": 0.10, "medium": 0.30, "hard": 0.60
        }
        
        # Load archetypes from config
        self.archetypes, self.proportions = load_archetypes(config_path)
        
        # Create branch pool
        self.branch_pool = [f"BR-{str(i).zfill(3)}" for i in range(1, num_branches + 1)]
        
        # Create employee pool (roughly 2 per branch)
        self.employee_pool = [f"EMP-{str(i).zfill(4)}" for i in range(1, num_branches * 2 + 1)]
        
        # Phase 1: Planning
        self.clients = self._create_population(num_clients)
        self.anomaly_plan = self._plan_anomalies()
    
    def _create_population(self, num_clients):
        clients = []
        for archetype_name, proportion in self.proportions.items():
            count = int(num_clients * proportion)
            archetype = self.archetypes[archetype_name]
            for _ in range(count):
                clients.append(Client(archetype, self.branch_pool))
        return clients
    
    def _plan_anomalies(self):
        plan = []
        
        # Estimate total events across simulation
        total_events = sum(
            sum(c.frequency.values()) * self.sim_days / 30
            for c in self.clients
        )
        
        num_anomalous_events = int(total_events * self.anomaly_rate)
        
        # Average daily events per client
        avg_daily_per_client = total_events / self.sim_days / len(self.clients)
        
        scenarios = {
            "easy": [
                {"name": "amount_spike", "actor": "client"},
                {"name": "round_amount", "actor": "client"},
                {"name": "duplicate_virement", "actor": "client"},
                {"name": "volume_spike", "actor": "employee"},
            ],
            "medium": [
                {"name": "frequency_burst", "actor": "client"},
                {"name": "cumulative_threshold", "actor": "client"},
                {"name": "timing_anomaly", "actor": "client"},
                {"name": "client_concentration", "actor": "employee"},
            ],
            "hard": [
                {"name": "smurfing", "actor": "client"},
                {"name": "cheque_structuring", "actor": "client"},
                {"name": "money_mule", "actor": "client"},
                {"name": "behavioral_drift", "actor": "client"},
                {"name": "cross_actor", "actor": "cross"},
            ],
        }
        
        durations = {"easy": 1, "medium": 5, "hard": 21}
        
        for difficulty, share in self.difficulty_distribution.items():
            target_events = int(num_anomalous_events * share)
            duration = durations[difficulty]
            
            # How many anomalous events one plan entry produces
            events_per_entry = max(1, duration * avg_daily_per_client)
            
            # How many plan entries we actually need
            n_entries = max(1, int(target_events / events_per_entry))
            
            available_scenarios = scenarios[difficulty]
            
            for _ in range(n_entries):
                scenario = np.random.choice(available_scenarios)
                client = np.random.choice(self.clients)
                start_day = np.random.randint(14, self.sim_days - 14)
                
                plan.append({
                    "client_id": client.client_id,
                    "scenario": scenario["name"],
                    "actor": scenario["actor"],
                    "difficulty": difficulty,
                    "start_day": start_day,
                    "end_day": start_day + duration,
                })
        
        return plan
    

    def generate(self, start_date="2025-01-01"):
        events = []
        start = datetime.date.fromisoformat(start_date)
        
        # Build a lookup for quick anomaly checking
        anomaly_lookup = {}
        for entry in self.anomaly_plan:
            cid = entry["client_id"]
            if cid not in anomaly_lookup:
                anomaly_lookup[cid] = []
            anomaly_lookup[cid].append(entry)
        
        # Tick through each day
        for day_offset in range(self.sim_days):
            current_date = start + datetime.timedelta(days=day_offset)
            day_of_month = current_date.day
            
            for client in self.clients:
                # Should this client transact today?
                # Compare day_of_month to client's preferred_day
                day_distance = abs(day_of_month - min(client.preferred_day, 
                    calendar.monthrange(current_date.year, current_date.month)[1]))
                
                # Higher probability near preferred day
                daily_prob = self._daily_probability(client, day_distance)
                
                if np.random.random() > daily_prob:
                    continue
                
                # How many operations today?
                total_monthly = sum(client.frequency.values())
                n_ops = max(1, np.random.poisson(total_monthly / 30))
                
                # Check if this client is in an anomaly window today
                is_anomaly = False
                active_scenario = None
                if client.client_id in anomaly_lookup:
                    for plan in anomaly_lookup[client.client_id]:
                        if plan["start_day"] <= day_offset <= plan["end_day"]:
                            is_anomaly = True
                            active_scenario = plan
                            break
                
                for _ in range(n_ops):
                    if is_anomaly:
                        event = self._generate_anomalous_event(client, current_date, active_scenario)
                    else:
                        event = self._generate_normal_event(client, current_date)
                    events.append(event)
        
        return events
    
    def _daily_probability(self, client, day_distance):
        # Bell curve around preferred day — closer = higher probability
        base_rate = sum(client.frequency.values()) / 30
        proximity_boost = np.exp(-0.5 * (day_distance / 3) ** 2)
        return min(base_rate * proximity_boost, 0.95)
    
    def _generate_normal_event(self, client, date):
        # Pick operation type from client's personal mix
        ops = list(client.operation_mix.keys())
        probs = list(client.operation_mix.values())
        op_type = np.random.choice(ops, p=probs)
        
        # Layer 3: sample amount with noise
        amount = np.clip(
            np.random.normal(client.amount_mean[op_type], client.amount_std[op_type]),
            5, None
        )
        amount = round(amount, 2)
        
        # Pick branch
        if np.random.random() < client.branch_loyalty:
            branch = client.home_branch
        else:
            branch = np.random.choice(self.branch_pool)
        
        # Pick employee from that branch (simplified)
        employee = np.random.choice(self.employee_pool)
        
        # Counterparty
        if op_type in ["virement", "cheque"]:
            if np.random.random() < client.counterparty_known_ratio:
                beneficiary = f"KNOWN-{np.random.randint(1, 6)}"
            else:
                beneficiary = f"NEW-{uuid.uuid4().hex[:8]}"
        else:
            beneficiary = None
        
        # Build payload based on operation type
        payload = self._build_payload(op_type, beneficiary)
        
        # Random hour during branch hours (8h-16h)
        hour = np.random.randint(8, 16)
        minute = np.random.randint(0, 60)
        timestamp = datetime.datetime.combine(date, datetime.time(hour, minute))
        
        return {
            "event_id": str(uuid.uuid4()),
            "client_id": client.client_id,
            "account_id": client.account_id,
            "employee_id": employee,
            "branch_id": branch,
            "timestamp": timestamp.isoformat(),
            "amount": amount,
            "currency": "TND",
            "channel": "guichet",
            "operation_type": op_type,
            "payload": json.dumps(payload),
            "is_anomaly": False,
            "anomaly_type": None,
        }
    
    def _build_payload(self, op_type, beneficiary=None):
        if op_type == "retrait":
            return {"mode": "especes"}
        elif op_type == "versement":
            return {"depositor_id": None}
        elif op_type == "virement":
            return {"beneficiary_id": beneficiary, "motif": "normal"}
        elif op_type == "cheque":
            return {"emitter_id": beneficiary, "cheque_number": f"CHQ-{uuid.uuid4().hex[:8]}"}
        return {}
    
    def _generate_anomalous_event(self, client, date, scenario):
        # Start with a normal event, then modify based on scenario
        event = self._generate_normal_event(client, date)
        event["is_anomaly"] = True
        event["anomaly_type"] = scenario["scenario"]
        
        if scenario["scenario"] == "amount_spike":
            event["amount"] = event["amount"] * np.random.uniform(5, 10)
        
        elif scenario["scenario"] == "round_amount":
            event["amount"] = np.random.choice([5000, 10000, 15000, 20000])
        
        elif scenario["scenario"] == "smurfing":
            event["operation_type"] = "versement"
            event["amount"] = np.random.uniform(8000, 9800)
            event["payload"] = json.dumps({"depositor_id": None})
        
        elif scenario["scenario"] == "frequency_burst":
            pass  # frequency is handled by n_ops in the main loop
        
        elif scenario["scenario"] == "timing_anomaly":
            hour = np.random.choice([6, 7, 17, 18])
            old_ts = datetime.datetime.fromisoformat(event["timestamp"])
            new_ts = old_ts.replace(hour=hour)
            event["timestamp"] = new_ts.isoformat()
        
        # ... more scenarios to implement later
        
        event["amount"] = round(event["amount"], 2)
        return event
    
    def export_csv(self, events, output_path="data/transactions.csv"):
        columns = [
            "event_id", "client_id", "account_id", "employee_id",
            "branch_id", "timestamp", "amount", "currency", "channel",
            "operation_type", "payload", "is_anomaly", "anomaly_type"
        ]
        
        # Sort by timestamp for temporal ordering
        events.sort(key=lambda e: e["timestamp"])
        
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(events)
        
        print(f"Exported {len(events)} events to {output_path}")