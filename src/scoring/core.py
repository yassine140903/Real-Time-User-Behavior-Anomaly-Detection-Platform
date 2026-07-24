# src/scoring/core.py

import json
import time
import redis
import numpy as np
from pathlib import Path
from src.scoring.ae_scorer import AEScorer
from src.scoring.lstm_scorer import LSTMScorer


class ScoringCore:
    EXPLAIN_THRESHOLD = 0.95

    def __init__(self, redis_client, fusion, models_dir="models", metrics=None):
        self.redis = redis_client
        self.fusion = fusion
        self.metrics = metrics

        background = np.load(Path(models_dir) / "shap_background.npy")
        self.ae_scorer = AEScorer(
            model=fusion.ae,
            scaler=fusion.scaler,
            feature_cols=fusion.feature_cols,
            background_data=background
        )
        self.lstm_scorer = LSTMScorer(
            model=fusion.lstm,
            scaler=fusion.scaler,
            feature_cols=fusion.feature_cols,
            seq_len=fusion.seq_len
        )

    # ── PUBLIC API ──────────────────────────────────────────

    def score_event(self, enriched: dict) -> dict:
        start_time = time.monotonic()

        try:
            cid = enriched["client_id"]

            # ── Cold-start guard ──────────────────────────────
            buf_raw = self.redis.get(f"sequence:client:{cid}")
            buf_len = len(json.loads(buf_raw)) if buf_raw else 0
            if buf_len < 5:
                output = self._build_cold_start_output(enriched, buf_len)
                if self.metrics:
                    op = enriched.get("operation_type", "unknown")
                    self.metrics.events_processed.labels(operation_type=op).inc()
                    self.metrics.fused_score.observe(0.5)
                return output

            # ── Normal scoring path ───────────────────────────
            features_scaled = self._extract_and_scale(enriched)
            sequence_scaled = self._fetch_and_scale_sequence(cid)
            days = enriched.get("account_age_days", 0)

            result = self.fusion.score(features_scaled, sequence_scaled, days)

            ae_explanation = None
            lstm_explanation = None
            if result["fused_score"] >= self.EXPLAIN_THRESHOLD:
                ae_explanation = self.ae_scorer.explain(features_scaled)
                if sequence_scaled is not None:
                    lstm_explanation = self.lstm_scorer.explain(
                        sequence_scaled[-self.fusion.seq_len:],
                        features_scaled
                    )

            output = self._build_output(enriched, result, ae_explanation, lstm_explanation)

            if self.metrics:
                op = enriched.get("operation_type", "unknown")
                self.metrics.events_processed.labels(operation_type=op).inc()
                self.metrics.fused_score.observe(result["fused_score"])

            return output

        except Exception as e:
            if self.metrics:
                error_class = "redis_failure" if isinstance(e, redis.RedisError) else "logic_error"
                self.metrics.errors.labels(error_class=error_class).inc()
            raise

        finally:
            if self.metrics:
                duration = time.monotonic() - start_time
                self.metrics.processing_duration.observe(duration)


    # ── PRIVATE ─────────────────────────────────────────────

    def _extract_and_scale(self, enriched):
        features = np.array([
            enriched.get(col, 0.0) for col in self.fusion.feature_cols
        ], dtype=np.float64)
        return self.fusion.scaler.transform(features.reshape(1, -1))[0]

    def _fetch_and_scale_sequence(self, client_id):
        data = self.redis.get(f"sequence:client:{client_id}")
        if not data:
            return None
        seq_raw = json.loads(data)
        if len(seq_raw) < self.fusion.seq_len:
            return None
        return self.fusion.scaler.transform(
            np.array(seq_raw[-self.fusion.seq_len:])
        )

    def _build_output(self, enriched, result, ae_explanation, lstm_explanation):
        return {
            # Pass-through identifiers
            "event_id": enriched["event_id"],
            "client_id": enriched["client_id"],
            "account_id": enriched["account_id"],
            "employee_id": enriched["employee_id"],
            "branch_id": enriched["branch_id"],
            "timestamp": enriched["timestamp"],
            "amount": enriched["amount"],
            "operation_type": enriched["operation_type"],

            # Pass-through features for Decision Service
            "z_amount": enriched.get("z_amount", 0.0),
            "tx_count_24h": enriched.get("tx_count_24h", 0),
            "cumulative_amount_24h": enriched.get("cumulative_amount_24h", 0.0),
            "near_threshold_count_7d": enriched.get("near_threshold_count_7d", 0),
            "same_employee_client_count_24h": enriched.get("same_employee_client_count_24h", 0),
            "is_new_counterparty": enriched.get("is_new_counterparty", 0),
            "is_round_amount": enriched.get("is_round_amount", 0),
            "has_duplicate_recent": enriched.get("has_duplicate_recent", 0),
            "account_age_days": enriched.get("account_age_days", 0),

            # Fusion scores
            **result,

            # Conditional explanations
            "ae_explanation": ae_explanation,
            "lstm_explanation": lstm_explanation,
        }

    def _build_cold_start_output(self, enriched, buf_len):
        return {
            # Pass-through identifiers
            "event_id": enriched["event_id"],
            "client_id": enriched["client_id"],
            "account_id": enriched["account_id"],
            "employee_id": enriched["employee_id"],
            "branch_id": enriched["branch_id"],
            "timestamp": enriched["timestamp"],
            "amount": enriched["amount"],
            "operation_type": enriched["operation_type"],

            # Pass-through features (still forwarded — DecisionCore may need them)
            "z_amount": enriched.get("z_amount", 0.0),
            "tx_count_24h": enriched.get("tx_count_24h", 0),
            "cumulative_amount_24h": enriched.get("cumulative_amount_24h", 0.0),
            "near_threshold_count_7d": enriched.get("near_threshold_count_7d", 0),
            "same_employee_client_count_24h": enriched.get("same_employee_client_count_24h", 0),
            "is_new_counterparty": enriched.get("is_new_counterparty", 0),
            "is_round_amount": enriched.get("is_round_amount", 0),
            "has_duplicate_recent": enriched.get("has_duplicate_recent", 0),
            "account_age_days": enriched.get("account_age_days", 0),

            # Neutral scores
            "ae_pct": 0.5,
            "lstm_pct": 0.5,
            "fused_score": 0.5,
            "w_lstm": 0.0,

            # No explanations
            "ae_explanation": None,
            "lstm_explanation": None,

            # Cold-start flag
            "cold_start": True,
            "buffer_length": buf_len,
        }