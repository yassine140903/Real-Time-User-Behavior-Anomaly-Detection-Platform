# src/streaming/decision_service.py

import json
import redis
from kafka import KafkaConsumer, KafkaProducer
from src.config import KAFKA_BOOTSTRAP, REDIS_HOST, REDIS_PORT
from src.decision.decision import DecisionService, DecisionInput

OPERATIONS = ["retrait", "versement", "virement", "cheque"]


class DecisionServiceStreaming:
    def __init__(self):
        # ── Kafka ───────────────────────────────────────────
        self.consumer = KafkaConsumer(
            'scored-events',
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id='decision-service',
            auto_offset_reset='earliest',
            value_deserializer=lambda v: json.loads(v.decode('utf-8'))
        )
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            key_serializer=lambda k: k.encode('utf-8'),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # ── Redis (for profile data) ───────────────────────
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

        # ── Stateless decision engine ──────────────────────
        self.engine = DecisionService()

    def get_client_profile(self, client_id):
        data = self.redis.get(f"profile:client:{client_id}")
        if data:
            return json.loads(data)
        return None

    def compute_expected_daily_rate(self, profile):
        """Sum all 30d op counts / 30 days."""
        total = 0
        for op in OPERATIONS:
            total += profile.get(f"tx_{op}_30d_count", 0)
        return max(total / 30.0, 0.1)  # floor at 0.1 to avoid div-by-zero

    def build_decision_input(self, scored):
        """Map scored event + Redis profile → DecisionInput."""
        cid = scored['client_id']
        profile = self.get_client_profile(cid)

        # ── Profile fields (with safe defaults) ─────────────
        if profile:
            archetype = profile.get('archetype', 'unknown')
            regulatory_client_type = profile.get('client_type', 'habitual')
            maturity_status = profile.get('maturity_status', 'mature')
            home_branch = profile.get('home_branch_id', '')
            expected_daily_rate = self.compute_expected_daily_rate(profile)
            days_since_opening = scored.get('account_age_days', 0)
        else:
            archetype = 'unknown'
            regulatory_client_type = 'habitual'
            maturity_status = 'mature'
            home_branch = ''
            expected_daily_rate = 1.0
            days_since_opening = 0

        return DecisionInput(
            # From scoring
            fused_score=scored['fused_score'],
            ae_pct=scored['ae_pct'],
            lstm_pct=scored.get('lstm_pct'),
            w_lstm=scored['w_lstm'],

            # From raw event (passed through)
            event_id=scored['event_id'],
            client_id=cid,
            employee_id=scored['employee_id'],
            branch_id=scored['branch_id'],
            operation_type=scored['operation_type'],
            amount=scored['amount'],
            timestamp=scored['timestamp'],

            # From enriched features (passed through)
            z_amount=scored.get('z_amount', 0.0),
            tx_count_24h=int(scored.get('tx_count_24h', 0)),
            cumulative_amount_24h=scored.get('cumulative_amount_24h', 0.0),
            near_threshold_count_7d=int(scored.get('near_threshold_count_7d', 0)),
            same_employee_client_count_24h=int(scored.get('same_employee_client_count_24h', 0)),
            is_new_beneficiary=bool(scored.get('is_new_counterparty', 0)),
            is_round_amount=bool(scored.get('is_round_amount', 0)),
            has_duplicate_recent=bool(scored.get('has_duplicate_recent', 0)),

            # From profile
            days_since_opening=days_since_opening,
            archetype=archetype,
            regulatory_client_type=regulatory_client_type,
            maturity_status=maturity_status,
            home_branch=home_branch,
            expected_daily_rate=expected_daily_rate,

            # Risk history (stubbed — no alerts table yet)
            alert_count_30d=0,
            confirmed_count_30d=0,
            rejected_count_30d=0,
            days_since_last_alert=999,

            # Explanations
            ae_explanation=scored.get('ae_explanation'),
            lstm_explanation=scored.get('lstm_explanation'),
        )

    def run(self):
        print("Decision Service started. Consuming from 'scored-events'...")
        count = 0
        alert_count = 0

        while True:
            messages = self.consumer.poll(timeout_ms=1000)
            if not messages:
                continue

            for topic_partition, batch in messages.items():
                for msg in batch:
                    scored = msg.value
                    inp = self.build_decision_input(scored)
                    decision = self.engine.decide(inp)

                    output = decision.to_dict()
                    output['client_id'] = scored['client_id']
                    output['account_id'] = scored['account_id']
                    output['employee_id'] = scored['employee_id']
                    output['branch_id'] = scored['branch_id']
                    output['timestamp'] = scored['timestamp']
                    output['amount'] = scored['amount']
                    output['operation_type'] = scored['operation_type']

                    self.producer.send(
                        'decisions',
                        key=scored['client_id'],
                        value=output
                    )
                    count += 1
                    if decision.tier.value > 0:  # anything above INFO
                        alert_count += 1

                    if count % 100 == 0:
                        self.producer.flush()
                        print(f"Decided {count} events | {alert_count} alerts")

    def close(self):
        self.consumer.close()
        self.producer.close()
        self.redis.close()


if __name__ == '__main__':
    service = DecisionServiceStreaming()
    try:
        service.run()
    except KeyboardInterrupt:
        print("\nDecision Service shutting down...")
    finally:
        service.close()