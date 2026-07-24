# src/decision/core.py

import json
import time
import redis
from src.decision.decision import DecisionService, DecisionInput, AlertTier

OPERATIONS = ["retrait", "versement", "virement", "cheque"]


class DecisionCore:
    def __init__(self, redis_client, engine: DecisionService, metrics=None):
        self.redis = redis_client
        self.engine = engine
        self.metrics = metrics
        with open("models/feature_cols.json") as f:
            self.feature_cols = json.load(f)

    # ── PUBLIC API ──────────────────────────────────────────

    def decide_event(self, scored: dict) -> dict:
        start_time = time.monotonic()

        try:
            profile = self._fetch_profile(scored["client_id"])
            inp = self._build_input(scored, profile)
            decision = self.engine.decide(inp)

            # ── Cold-start tier cap ───────────────────────
            if scored.get("cold_start", False):
                buf_len = scored.get("buffer_length", 0)
                decision.reasons.append(
                    f"Cold-start client ({buf_len}/5 events) — tier capped at REVIEW"
                )
                if decision.tier > AlertTier.REVIEW:
                    decision.tier = AlertTier.REVIEW

            output = self._build_output(scored, decision)

            if self.metrics:
                op = scored.get("operation_type", "unknown")
                self.metrics.events_processed.labels(operation_type=op).inc()

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

    def _fetch_profile(self, client_id: str):
        data = self.redis.get(f"profile:client:{client_id}")
        return json.loads(data) if data else None

    def _compute_expected_daily_rate(self, profile):
        total = sum(profile.get(f"tx_{op}_30d_count", 0) for op in OPERATIONS)
        return max(total / 30.0, 1.0)

    def _build_input(self, scored, profile):
        if profile:
            archetype = profile.get("archetype", "unknown")
            regulatory_client_type = profile.get("client_type", "habitual")
            maturity_status = profile.get("maturity_status", "mature")
            home_branch = profile.get("home_branch_id", "")
            expected_daily_rate = self._compute_expected_daily_rate(profile)
            days_since_opening = scored.get("account_age_days", 0)
        else:
            archetype = "unknown"
            regulatory_client_type = "habitual"
            maturity_status = "mature"
            home_branch = ""
            expected_daily_rate = 1.0
            days_since_opening = 0

        return DecisionInput(
            fused_score=scored["fused_score"],
            ae_pct=scored["ae_pct"],
            lstm_pct=scored.get("lstm_pct"),
            w_lstm=scored["w_lstm"],

            event_id=scored["event_id"],
            client_id=scored["client_id"],
            employee_id=scored["employee_id"],
            branch_id=scored["branch_id"],
            operation_type=scored["operation_type"],
            amount=scored["amount"],
            timestamp=scored["timestamp"],

            z_amount=scored.get("z_amount", 0.0),
            tx_count_24h=int(scored.get("tx_count_24h", 0)),
            cumulative_amount_24h=scored.get("cumulative_amount_24h", 0.0),
            near_threshold_count_7d=int(scored.get("near_threshold_count_7d", 0)),
            same_employee_client_count_24h=int(scored.get("same_employee_client_count_24h", 0)),
            is_new_beneficiary=bool(scored.get("is_new_counterparty", 0)),
            is_round_amount=bool(scored.get("is_round_amount", 0)),
            has_duplicate_recent=bool(scored.get("has_duplicate_recent", 0)),

            days_since_opening=days_since_opening,
            archetype=archetype,
            regulatory_client_type=regulatory_client_type,
            maturity_status=maturity_status,
            home_branch=home_branch,
            expected_daily_rate=expected_daily_rate,

            alert_count_30d=0,
            confirmed_count_30d=0,
            rejected_count_30d=0,
            days_since_last_alert=999,

            ae_explanation=scored.get("ae_explanation"),
            lstm_explanation=scored.get("lstm_explanation"),
        )

    def _build_output(self, scored, decision):
        output = decision.to_dict()
        output["client_id"] = scored["client_id"]
        output["account_id"] = scored["account_id"]
        output["employee_id"] = scored["employee_id"]
        output["branch_id"] = scored["branch_id"]
        output["timestamp"] = scored["timestamp"]
        output["amount"] = scored["amount"]
        output["operation_type"] = scored["operation_type"]
        output["ae_explanation"] = scored.get("ae_explanation")
        output["lstm_explanation"] = scored.get("lstm_explanation")
        output["enriched_features"] = {col: scored.get(col, 0.0) for col in self.feature_cols}
        return output