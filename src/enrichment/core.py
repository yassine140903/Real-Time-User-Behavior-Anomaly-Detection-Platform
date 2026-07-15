# src/enrichment/core.py

import json
import numpy as np
import pandas as pd
import redis
from pathlib import Path
import time

OPS = ["retrait", "versement", "virement", "cheque"]
ARCHETYPES = ["salaried", "small_business", "student", "retiree", "big_business"]
BUFFER_SIZE = 50
EMP_BUFFER_SIZE = 100


class EnrichmentCore:
    def __init__(self, redis_client: redis.Redis, models_dir: str = "models",
                 metrics=None):
        self.redis = redis_client
        self.metrics = metrics
        models_path = Path(models_dir)
        with open(models_path / "feature_cols.json") as f:
            self.feature_cols = json.load(f)
        with open(models_path / "lstm_config.json") as f:
            self.seq_len = json.load(f)["seq_len"]

    # ── PUBLIC API (single entry point) ─────────────────────────

    def enrich_event(self, event: dict, dry_run: bool = False) -> dict:
        start_time = time.monotonic()

        try:
            cid = event["client_id"]
            eid = event["employee_id"]
            ts = pd.Timestamp(event["timestamp"])
            amount = float(event["amount"])
            op = event["operation_type"]

            profile = self._fetch_profile(cid)

            if profile is None and self.metrics:
                self.metrics.errors.labels(error_class="data_gap").inc()

            cbuf, ebuf = self._fetch_buffers(cid, eid)

            arch = profile.get("archetype", "salaried") if profile else "salaried"
            baseline_raw = self.redis.get(f"baseline:{arch}")
            archetype_baseline = json.loads(baseline_raw) if baseline_raw else None

            features = self._compute_features(event, profile, cbuf, ebuf, ts, amount, op, eid,
                                               archetype_baseline=archetype_baseline)

            features["event_id"] = event["event_id"]
            features["client_id"] = cid
            features["account_id"] = event["account_id"]
            features["employee_id"] = eid
            features["branch_id"] = event["branch_id"]
            features["timestamp"] = event["timestamp"]
            features["amount"] = amount
            features["operation_type"] = op
            features["payload"] = event["payload"]

            if not dry_run:
                self._update_buffers(cid, eid, ts, amount, op, eid, event["payload"], cbuf, ebuf)
                self._update_sequence(cid, features)

            # Success
            if self.metrics:
                self.metrics.events_processed.labels(operation_type=op).inc()

            return features

        except redis.RedisError:
            if self.metrics:
                self.metrics.errors.labels(error_class="redis_failure").inc()
            raise

        except Exception:
            if self.metrics:
                self.metrics.errors.labels(error_class="logic_error").inc()
            raise

        finally:
            if self.metrics:
                duration = time.monotonic() - start_time
                self.metrics.processing_duration.observe(duration)

    # ── PRIVATE: Data Access ────────────────────────────────────

    def _fetch_profile(self, client_id: str):
        data = self.redis.get(f"profile:client:{client_id}")
        return json.loads(data) if data else None

    def _fetch_buffers(self, client_id: str, employee_id: str):
        cbuf_raw = self.redis.get(f"buffer:client:{client_id}")
        ebuf_raw = self.redis.get(f"buffer:employee:{employee_id}")

        cbuf = json.loads(cbuf_raw) if cbuf_raw else []
        ebuf = json.loads(ebuf_raw) if ebuf_raw else []

        # Parse timestamps for comparison
        for e in cbuf:
            e["ts"] = pd.Timestamp(e["ts"])
        for e in ebuf:
            e["ts"] = pd.Timestamp(e["ts"])

        return cbuf, ebuf

    def _update_buffers(self, cid, eid, ts, amount, op, employee_id, payload, cbuf, ebuf):
        # Append new entries
        cbuf.append({"ts": ts, "amount": amount, "op": op, "eid": employee_id, "payload": payload})
        if len(cbuf) > BUFFER_SIZE:
            cbuf = cbuf[-BUFFER_SIZE:]

        ebuf.append({"ts": ts, "cid": cid})
        if len(ebuf) > EMP_BUFFER_SIZE:
            ebuf = ebuf[-EMP_BUFFER_SIZE:]

        # Serialize timestamps and write back
        cbuf_store = [
            {**e, "ts": e["ts"].isoformat() if isinstance(e["ts"], pd.Timestamp) else e["ts"]}
            for e in cbuf
        ]
        ebuf_store = [
            {**e, "ts": e["ts"].isoformat() if isinstance(e["ts"], pd.Timestamp) else e["ts"]}
            for e in ebuf
        ]

        self.redis.set(f"buffer:client:{cid}", json.dumps(cbuf_store))
        self.redis.set(f"buffer:employee:{eid}", json.dumps(ebuf_store))

    def _update_sequence(self, client_id: str, features: dict):
        feat_vector = [features.get(col, 0.0) for col in self.feature_cols]
        seq_key = f"sequence:client:{client_id}"

        raw = self.redis.get(seq_key)
        seq_buf = json.loads(raw) if raw else []
        seq_buf.append(feat_vector)
        if len(seq_buf) > self.seq_len:
            seq_buf = seq_buf[-self.seq_len:]

        self.redis.set(seq_key, json.dumps(seq_buf))

    # ── PRIVATE: Feature Computation ───────────────────────────

    def _compute_features(self, row, prof, cbuf, ebuf, ts, amount, op, eid, archetype_baseline=None):
        f = {}

        # ── RAW EVENT ──────────────────────────────────────────
        f["amount"] = amount
        f["hour"] = ts.hour

        for o in OPS:
            f[f"op_{o}"] = 1 if op == o else 0

        if prof is not None:
            f["is_home_branch"] = 1 if row["branch_id"] == prof.get("home_branch_id") else 0
        else:
            f["is_home_branch"] = 1

        # ── AMOUNT CONTRAST ────────────────────────────────────
        if prof is not None:
            p_count = prof.get(f"tx_{op}_30d_count", 0)
            if p_count > 0:
                p_mean = float(prof.get(f"tx_{op}_30d_mean", amount))
                p_std = float(prof.get(f"tx_{op}_30d_std", 1.0))
            else:
                p_count_90 = prof.get(f"tx_{op}_90d_count", 0)
                if p_count_90 > 0:
                    p_mean = float(prof.get(f"tx_{op}_90d_mean", amount))
                    p_std = float(prof.get(f"tx_{op}_90d_std", 1.0))
                else:
                    if archetype_baseline:
                        p_mean = float(archetype_baseline.get(f"amount_{op}_mean", amount))
                        p_std = float(archetype_baseline.get(f"amount_{op}_std", 1.0))
                    else:
                        p_mean = amount
                        p_std = 1.0

            p_mean = float(p_mean) if p_mean is not None else amount
            p_std = float(p_std) if p_std is not None else 1.0
            f["z_amount"] = (amount - p_mean) / max(p_std, 1.0)
            f["amount_to_mean_ratio"] = amount / max(p_mean, 1.0)
        else:
            f["z_amount"] = 0.0
            f["amount_to_mean_ratio"] = 1.0

        f["is_above_threshold"] = 1 if amount > 10000 else 0
        f["is_near_threshold"] = 1 if 8000 <= amount <= 9999 else 0
        f["is_round_amount"] = 1 if amount >= 1000 and abs(amount % 1000) < 1 else 0
        f["is_near_cheque_ceiling"] = 1 if 25000 <= amount <= 30000 else 0

        # ── OPERATION CONTRAST ─────────────────────────────────
        if prof is not None:
            op_dist = prof.get("dist_30d_operation", {})
            f["op_type_probability"] = float(op_dist.get(op, 0.25))
        else:
            f["op_type_probability"] = 0.25

        # ── TIMING CONTRAST ────────────────────────────────────
        f["is_outside_hours"] = 1 if ts.hour < 8 or ts.hour >= 16 else 0
        f["day_distance_from_preferred"] = 2

        # ── COUNTERPARTY CONTRAST ──────────────────────────────
        try:
            payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
        except (json.JSONDecodeError, TypeError):
            payload = {}

        if op == "virement":
            ben = payload.get("beneficiary_id", "")
            f["is_new_counterparty"] = 1 if ben and str(ben).startswith("NEW-") else 0
        elif op == "cheque":
            emt = payload.get("emitter_id", "")
            f["is_new_counterparty"] = 1 if emt and str(emt).startswith("NEW-") else 0
        else:
            f["is_new_counterparty"] = 0

        # ── VELOCITY FROM CLIENT BUFFER ────────────────────────
        t_24h = ts - pd.Timedelta(hours=24)
        t_7d = ts - pd.Timedelta(days=7)

        recent_24h = [e for e in cbuf if e["ts"] >= t_24h]
        recent_7d = [e for e in cbuf if e["ts"] >= t_7d]

        f["tx_count_24h"] = len(recent_24h)
        f["tx_count_7d"] = len(recent_7d)
        f["cumulative_amount_24h"] = sum(e["amount"] for e in recent_24h)

        has_dup = 0
        for e in recent_24h:
            if e["op"] == op and abs(e["amount"] - amount) / max(amount, 1) < 0.10:
                has_dup = 1
                break
        f["has_duplicate_recent"] = has_dup

        f["near_threshold_count_7d"] = sum(
            1 for e in recent_7d if 8000 <= e["amount"] <= 9999
        )

        # ── EMPLOYEE FROM BUFFER ───────────────────────────────
        emp_24h = [e for e in ebuf if e["ts"] >= t_24h]
        f["employee_tx_count_24h"] = len(emp_24h)

        f["same_employee_client_count_24h"] = sum(
            1 for e in cbuf if e["ts"] >= t_24h and e.get("eid") == eid
        )

        # ── CONTEXT ────────────────────────────────────────────
        if prof is not None:
            arch = prof.get("archetype", "salaried")
            for a in ARCHETYPES:
                f[f"arch_{a}"] = 1 if arch == a else 0
            f["account_age_days"] = int(prof.get("account_age_days", 0))
        else:
            for a in ARCHETYPES:
                f[f"arch_{a}"] = 0
            f["account_age_days"] = 0

        return f