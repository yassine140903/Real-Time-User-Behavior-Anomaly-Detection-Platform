# src/scoring/core.py

import json
import numpy as np
from pathlib import Path
from src.scoring.ae_scorer import AEScorer
from src.scoring.lstm_scorer import LSTMScorer


class ScoringCore:
    EXPLAIN_THRESHOLD = 0.95

    def __init__(self, redis_client, fusion, models_dir="models"):
        self.redis = redis_client
        self.fusion = fusion

        # Create scorers internally — tightly coupled to fusion
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
        cid = enriched["client_id"]

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

        return self._build_output(enriched, result, ae_explanation, lstm_explanation)

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