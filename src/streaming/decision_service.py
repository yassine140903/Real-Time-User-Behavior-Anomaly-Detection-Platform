# src/streaming/decision_service.py

import json
from kafka import KafkaConsumer, KafkaProducer
from src.config import KAFKA_BOOTSTRAP
from src.decision import DecisionCore


class DecisionStreamingService:
    def __init__(self, core: DecisionCore):
        self.core = core

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
                    output = self.core.decide_event(scored)

                    self.producer.send(
                        'decisions',
                        key=scored['client_id'],
                        value=output
                    )

                    count += 1
                    if output["tier"] != "INFO":
                        alert_count += 1

                    if count % 100 == 0:
                        self.producer.flush()
                        print(f"Decided {count} events | {alert_count} alerts")

    def close(self):
        self.consumer.close()
        self.producer.close()


if __name__ == '__main__':
    import redis
    from src.config import REDIS_HOST, REDIS_PORT
    from src.decision.decision import DecisionService

    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    engine = DecisionService()
    core = DecisionCore(redis_client, engine)
    service = DecisionStreamingService(core)

    try:
        service.run()
    except KeyboardInterrupt:
        print("\nDecision Service shutting down...")
    finally:
        service.close()
        redis_client.close()