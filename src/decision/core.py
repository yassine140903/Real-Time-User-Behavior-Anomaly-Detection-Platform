# src/decision/core.py

import json
from src.decision.decision import DecisionService, DecisionInput

OPERATIONS = ["retrait", "versement", "virement", "cheque"]


class DecisionCore:
    def __init__(self, redis_client, engine: DecisionService):
        self.redis = redis_client
        self.engine = engine

    # ── PUBLIC API ──────────────────────────────────────────

    def decide_event(self, scored: dict) -> dict:
        profile = self._fetch_profile(scored["client_id"])
        inp = self._build_input(scored, profile)
        decision = self.engine.decide(inp)
        return self._build_output(scored, decision)

    # ── PRIVATE ─────────────────────────────────────────────

    def _fetch_profile(self, client_id: str):
        data = self.redis.get(f"profile:client:{client_id}")
        return json.loads(data) if data else None

    def _compute_expected_daily_rate(self, profile):
        total = sum(profile.get(f"tx_{op}_30d_count", 0) for op in OPERATIONS)
        return max(total / 30.0, 0.1)

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
        return output