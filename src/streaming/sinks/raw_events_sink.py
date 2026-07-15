# src/streaming/sinks/raw_events_sink.py

import json
import psycopg2
import psycopg2.extras
from kafka import KafkaConsumer, KafkaProducer
from src.config import KAFKA_BOOTSTRAP, DB_CONFIG
import time


class RawEventsSink:
    def __init__(self, db_config, metrics=None):
        self.metrics = metrics

        self.consumer = KafkaConsumer(
            'raw-events',
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id='raw-events-sink',
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
        print("Raw Events Sink started. Consuming from 'raw-events'...")
        count = 0

        while True:
            messages = self.consumer.poll(timeout_ms=1000)
            if not messages:
                continue

            for topic_partition, batch in messages.items():
                for msg in batch:
                    event = msg.value
                    self._persist(event)

                    count += 1
                    if count % 100 == 0:
                        print(f"Persisted {count} events to transactions")

    def _persist(self, event):
        start = time.monotonic()

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO transactions 
                    (event_id, client_id, account_id, employee_id, branch_id,
                     timestamp, amount, currency, channel, operation_type, payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
            """, (
                event['event_id'],
                event['client_id'],
                event['account_id'],
                event['employee_id'],
                event['branch_id'],
                event['timestamp'],
                event['amount'],
                event.get('currency', 'TND'),
                event.get('channel', 'guichet'),
                event['operation_type'],
                event.get('payload') if isinstance(event.get('payload'), str)
                    else json.dumps(event.get('payload'))
            ))
            cursor.close()

            if self.metrics:
                op = event.get('operation_type', 'unknown')
                self.metrics.events_processed.labels(operation_type=op).inc()

        except (psycopg2.IntegrityError, psycopg2.errors.ForeignKeyViolation) as e:
            self.conn.rollback()
            self._send_to_dlq(event, str(e))

            if self.metrics:
                self.metrics.errors.labels(error_class='fk_violation').inc()

        except Exception as e:
            self.conn.rollback()
            self._send_to_dlq(event, str(e))

            if self.metrics:
                self.metrics.errors.labels(error_class='db_error').inc()

        finally:
            if self.metrics:
                duration = time.monotonic() - start
                self.metrics.processing_duration.observe(duration)

    def _send_to_dlq(self, event, error_reason):
        dlq_message = {
            'original_event': event,
            'error': error_reason,
            'source': 'raw-events-sink'
        }
        try:
            self.producer.send('dlq-raw-events', value=dlq_message)
            self.producer.flush()
            print(f"Event {event.get('event_id')} sent to DLQ: {error_reason}")
        except Exception as dlq_error:
            print(f"CRITICAL: DLQ publish failed for {event.get('event_id')}: {dlq_error}")

    def close(self):
        self.consumer.close()
        self.producer.close()
        self.conn.close()


if __name__ == '__main__':
    from src.monitoring.metrics import ServiceMetrics

    metrics = ServiceMetrics("raw_events_sink")
    metrics.start_server(port=9103)

    sink = RawEventsSink(db_config=DB_CONFIG, metrics=metrics)

    try:
        sink.run()
    except KeyboardInterrupt:
        print("\nRaw Events Sink shutting down...")
    finally:
        sink.close()