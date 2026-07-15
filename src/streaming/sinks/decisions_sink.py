# src/streaming/sinks/decisions_sink.py

import json
import time
import redis
import psycopg2
import psycopg2.extras
from kafka import KafkaConsumer, KafkaProducer
from src.config import KAFKA_BOOTSTRAP, DB_CONFIG, REDIS_HOST, REDIS_PORT


class DecisionsSink:
    def __init__(self, db_config, redis_client, metrics=None):
        self.metrics = metrics
        self.redis = redis_client

        self.consumer = KafkaConsumer(
            'decisions',
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id='decisions-sink',
            auto_offset_reset='earliest',
            value_deserializer=lambda v: json.loads(v.decode('utf-8'))
        )

        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        self.conn = psycopg2.connect(**db_config)
        self.conn.autocommit = True

    def run(self):
        print("Decisions Sink started. Consuming from 'decisions'...")
        count = 0

        while True:
            messages = self.consumer.poll(timeout_ms=1000)
            if not messages:
                continue

            for topic_partition, batch in messages.items():
                for msg in batch:
                    decision = msg.value
                    self._process(decision)

                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count} decisions")

    def _process(self, decision):
        start = time.monotonic()

        try:
            self._persist_alert(decision)
            self._update_risk_history(decision)

            if self.metrics:
                op = decision.get('operation_type', 'unknown')
                self.metrics.events_processed.labels(operation_type=op).inc()

        except Exception as e:
            if self.metrics:
                error_class = self._classify_error(e)
                self.metrics.errors.labels(error_class=error_class).inc()
            print(f"ERROR processing decision {decision.get('event_id')}: {e}")

        finally:
            if self.metrics:
                duration = time.monotonic() - start
                self.metrics.processing_duration.observe(duration)

    # ── PostgreSQL: persist alert ───────────────────────────

    def _persist_alert(self, decision):
        try:
            print(f"INSERTING alert: {decision.get('event_id')}")
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO alerts
                    (event_id, client_id, employee_id, branch_id,
                     timestamp, amount, operation_type, tier, fused_score,
                     reasons, regulatory_flags, explanation)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
            """, (
                decision['event_id'],
                decision['client_id'],
                decision['employee_id'],
                decision['branch_id'],
                decision['timestamp'],
                decision['amount'],
                decision['operation_type'],
                decision['tier'],
                decision['fused_score'],
                json.dumps(decision.get('reasons', [])),
                json.dumps(decision.get('regulatory_flags', [])),
                json.dumps(decision.get('explanation')),
            ))
            print(f"INSERT OK, rowcount={cursor.rowcount}")
            cursor.close()

        except psycopg2.Error as e:
            self.conn.rollback()
            self._send_to_dlq(decision, str(e))

    # ── Redis: update risk history ──────────────────────────

    def _update_risk_history(self, decision):
        tier = decision.get('tier', 'INFO')
        if tier == 'INFO':
            return

        client_id = decision['client_id']
        profile_key = f"profile:client:{client_id}"

        try:
            data = self.redis.get(profile_key)
            if not data:
                return

            profile = json.loads(data)

            # Increment tier-specific counters
            tier_lower = tier.lower()
            count_key = f"alert_{tier_lower}_30d_count"
            profile[count_key] = profile.get(count_key, 0) + 1

            # Update total alert count
            profile["alert_count_30d"] = profile.get("alert_count_30d", 0) + 1

            # Update last alert timestamp
            profile["last_alert_timestamp"] = decision['timestamp']

            self.redis.set(profile_key, json.dumps(profile))

        except redis.RedisError as e:
            # Redis failure doesn't warrant DLQ — pipeline already degraded
            print(f"WARNING: Redis update failed for {client_id}: {e}")

    # ── DLQ ─────────────────────────────────────────────────

    def _send_to_dlq(self, decision, error_reason):
        dlq_message = {
            'original_event': decision,
            'error': error_reason,
            'source': 'decisions-sink'
        }
        try:
            self.producer.send('dlq-decisions', value=dlq_message)
            self.producer.flush()
            print(f"Decision {decision.get('event_id')} sent to DLQ: {error_reason}")
        except Exception as dlq_error:
            print(f"CRITICAL: DLQ publish failed for {decision.get('event_id')}: {dlq_error}")

    # ── Helpers ─────────────────────────────────────────────

    def _classify_error(self, e):
        if isinstance(e, psycopg2.Error):
            return 'db_error'
        if isinstance(e, redis.RedisError):
            return 'redis_failure'
        return 'logic_error'

    def close(self):
        self.consumer.close()
        self.producer.close()
        self.conn.close()


if __name__ == '__main__':
    from src.monitoring.metrics import ServiceMetrics

    metrics = ServiceMetrics("decisions_sink")
    metrics.start_server(port=9104)

    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    sink = DecisionsSink(db_config=DB_CONFIG, redis_client=redis_client, metrics=metrics)

    try:
        sink.run()
    except KeyboardInterrupt:
        print("\nDecisions Sink shutting down...")
    finally:
        sink.close()
        redis_client.close()