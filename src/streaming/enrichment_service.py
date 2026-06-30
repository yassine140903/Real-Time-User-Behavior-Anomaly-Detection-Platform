# src/streaming/enrichment_service.py

import json
import time
import redis
import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from src.config import KAFKA_BOOTSTRAP, REDIS_HOST, REDIS_PORT
from pathlib import Path

OPS = ["retrait", "versement", "virement", "cheque"]
ARCHETYPES = ["salaried", "small_business", "student", "retiree", "big_business"]
BUFFER_SIZE = 50
EMP_BUFFER_SIZE = 100


class EnrichmentService:
    def __init__(self):
        self.consumer = KafkaConsumer(
            'raw-events',
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id='enrichment-service',
            auto_offset_reset='earliest',
            value_deserializer=lambda v: json.loads(v.decode('utf-8'))
        )

        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            key_serializer=lambda k: k.encode('utf-8'),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        models_dir = Path("models")
        with open(models_dir / "feature_cols.json") as f:
            self.feature_cols = json.load(f)
        with open(models_dir / "lstm_config.json") as f:
            self.seq_len = json.load(f)["seq_len"]

    def get_client_profile(self, client_id):
        data = self.redis.get(f"profile:client:{client_id}")
        if data:
            return json.loads(data)
        return None

    def get_buffer(self, key):
        data = self.redis.get(key)
        if data:
            return json.loads(data)
        return []

    def set_buffer(self, key, buffer_list):
        self.redis.set(key, json.dumps(buffer_list))

    def enrich_event(self, event):
        cid = event['client_id']
        eid = event['employee_id']
        ts = pd.Timestamp(event['timestamp'])
        amount = float(event['amount'])
        op = event['operation_type']

        # Fetch profile and buffers from Redis
        prof = self.get_client_profile(cid)
        cbuf = self.get_buffer(f"buffer:client:{cid}")
        ebuf = self.get_buffer(f"buffer:employee:{eid}")

        # Convert buffer timestamps back to comparable format
        for e in cbuf:
            e['ts'] = pd.Timestamp(e['ts'])
        for e in ebuf:
            e['ts'] = pd.Timestamp(e['ts'])

        # Compute features — same logic as batch
        f = self._compute_features(event, prof, cbuf, ebuf, ts, amount, op, eid)

        # Attach identifiers for downstream services
        f['event_id'] = event['event_id']
        f['client_id'] = cid
        f['account_id'] = event['account_id']
        f['employee_id'] = eid
        f['branch_id'] = event['branch_id']
        f['timestamp'] = event['timestamp']
        f['amount'] = amount
        f['operation_type'] = op
        f['payload'] = event['payload']

        # Update buffers AFTER feature computation
        cbuf.append({
            'ts': ts.isoformat(),
            'amount': amount,
            'op': op,
            'eid': eid,
            'payload': event['payload']
        })
        if len(cbuf) > BUFFER_SIZE:
            cbuf = cbuf[-BUFFER_SIZE:]

        ebuf.append({
            'ts': ts.isoformat(),
            'cid': cid
        })
        if len(ebuf) > EMP_BUFFER_SIZE:
            ebuf = ebuf[-EMP_BUFFER_SIZE:]

        # Write updated buffers back to Redis
        # Serialize timestamps for storage
        cbuf_store = []
        for e in cbuf:
            entry = dict(e)
            if isinstance(entry['ts'], pd.Timestamp):
                entry['ts'] = entry['ts'].isoformat()
            cbuf_store.append(entry)

        ebuf_store = []
        for e in ebuf:
            entry = dict(e)
            if isinstance(entry['ts'], pd.Timestamp):
                entry['ts'] = entry['ts'].isoformat()
            ebuf_store.append(entry)

        self.set_buffer(f"buffer:client:{cid}", cbuf_store)
        self.set_buffer(f"buffer:employee:{eid}", ebuf_store)

        feat_vector = [f.get(col, 0.0) for col in self.feature_cols]
        seq_key = f"sequence:client:{cid}"
        seq_buf = self.get_buffer(seq_key)
        seq_buf.append(feat_vector)
        if len(seq_buf) > self.seq_len:
            seq_buf = seq_buf[-self.seq_len:]
        self.set_buffer(seq_key, seq_buf)

        return f

    def _compute_features(self, row, prof, cbuf, ebuf, ts, amount, op, eid):
        f = {}

        # ── RAW EVENT ──────────────────────────────────────────
        f['amount'] = amount
        f['hour'] = ts.hour

        for o in OPS:
            f[f'op_{o}'] = 1 if op == o else 0

        if prof is not None:
            f['is_home_branch'] = 1 if row['branch_id'] == prof.get('home_branch_id') else 0
        else:
            f['is_home_branch'] = 1

        # ── AMOUNT CONTRAST ────────────────────────────────────
        if prof is not None:
            p_count = prof.get(f'tx_{op}_30d_count', 0)
            if p_count > 0:
                p_mean = float(prof.get(f'tx_{op}_30d_mean', amount))
                p_std = float(prof.get(f'tx_{op}_30d_std', 1.0))
            else:
                p_count_90 = prof.get(f'tx_{op}_90d_count', 0)
                if p_count_90 > 0:
                    p_mean = float(prof.get(f'tx_{op}_90d_mean', amount))
                    p_std = float(prof.get(f'tx_{op}_90d_std', 1.0))
                else:
                    arch = prof.get('archetype', 'salaried')
                    baseline_data = self.redis.get(f"baseline:{arch}")
                    if baseline_data:
                        baseline = json.loads(baseline_data)
                        p_mean = float(baseline.get(f'amount_{op}_mean', amount))
                        p_std = float(baseline.get(f'amount_{op}_std', 1.0))
                    else:
                        p_mean = amount
                        p_std = 1.0





            p_mean = float(p_mean) if p_mean is not None else amount
            p_std = float(p_std) if p_std is not None else 1.0
            f['z_amount'] = (amount - p_mean) / max(p_std, 1.0)
            f['amount_to_mean_ratio'] = amount / max(p_mean, 1.0)
        else:
            f['z_amount'] = 0.0
            f['amount_to_mean_ratio'] = 1.0

        f['is_above_threshold'] = 1 if amount > 10000 else 0
        f['is_near_threshold'] = 1 if 8000 <= amount <= 9999 else 0
        f['is_round_amount'] = 1 if amount >= 1000 and abs(amount % 1000) < 1 else 0
        f['is_near_cheque_ceiling'] = 1 if 25000 <= amount <= 30000 else 0

        # ── OPERATION CONTRAST ─────────────────────────────────
        if prof is not None:
            op_dist = prof.get('dist_30d_operation', {})
            f['op_type_probability'] = float(op_dist.get(op, 0.25))
        else:
            f['op_type_probability'] = 0.25

        # ── TIMING CONTRAST ────────────────────────────────────
        f['is_outside_hours'] = 1 if ts.hour < 8 or ts.hour >= 16 else 0

        f['day_distance_from_preferred'] = 2

        # ── COUNTERPARTY CONTRAST ──────────────────────────────
        try:
            payload = json.loads(row['payload']) if isinstance(row['payload'], str) else row['payload']
        except (json.JSONDecodeError, TypeError):
            payload = {}

        if op == 'virement':
            ben = payload.get('beneficiary_id', '')
            f['is_new_counterparty'] = 1 if ben and str(ben).startswith('NEW-') else 0
        elif op == 'cheque':
            emt = payload.get('emitter_id', '')
            f['is_new_counterparty'] = 1 if emt and str(emt).startswith('NEW-') else 0
        else:
            f['is_new_counterparty'] = 0

        # ── VELOCITY FROM CLIENT BUFFER ────────────────────────
        t_24h = ts - pd.Timedelta(hours=24)
        t_7d = ts - pd.Timedelta(days=7)

        recent_24h = [e for e in cbuf if e['ts'] >= t_24h]
        recent_7d = [e for e in cbuf if e['ts'] >= t_7d]

        f['tx_count_24h'] = len(recent_24h)
        f['tx_count_7d'] = len(recent_7d)
        f['cumulative_amount_24h'] = sum(e['amount'] for e in recent_24h)

        has_dup = 0
        for e in recent_24h:
            if e['op'] == op and abs(e['amount'] - amount) / max(amount, 1) < 0.10:
                has_dup = 1
                break
        f['has_duplicate_recent'] = has_dup

        f['near_threshold_count_7d'] = sum(
            1 for e in recent_7d if 8000 <= e['amount'] <= 9999
        )

        # ── EMPLOYEE FROM BUFFER ───────────────────────────────
        emp_24h = [e for e in ebuf if e['ts'] >= t_24h]
        f['employee_tx_count_24h'] = len(emp_24h)

        f['same_employee_client_count_24h'] = sum(
            1 for e in cbuf if e['ts'] >= t_24h and e.get('eid') == eid
        )

        # ── CONTEXT ────────────────────────────────────────────
        if prof is not None:
            arch = prof.get('archetype', 'salaried')
            for a in ARCHETYPES:
                f[f'arch_{a}'] = 1 if arch == a else 0
            f['account_age_days'] = int(prof.get('account_age_days', 0))
        else:
            for a in ARCHETYPES:
                f[f'arch_{a}'] = 0
            f['account_age_days'] = 0

        return f

    def run(self):
        print("Enrichment Service started. Consuming from 'raw-events'...")
        count = 0

        while True:
            messages = self.consumer.poll(timeout_ms=1000)
            if not messages:
                continue

            for topic_partition, batch in messages.items():
                for msg in batch:
                    event = msg.value
                    enriched = self.enrich_event(event)

                    self.producer.send(
                        'enriched-events',
                        key=event['client_id'],
                        value=enriched
                    )

                    count += 1
                    if count % 100 == 0:
                        self.producer.flush()
                        print(f"Enriched {count} events")

    def close(self):
        self.consumer.close()
        self.producer.close()
        self.redis.close()


if __name__ == '__main__':
    service = EnrichmentService()
    try:
        service.run()
    except KeyboardInterrupt:
        print(f"\nEnrichment Service shutting down...")
    finally:
        service.close()