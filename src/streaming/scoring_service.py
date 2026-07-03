# src/streaming/scoring_service.py

import json
from kafka import KafkaConsumer, KafkaProducer
from src.config import KAFKA_BOOTSTRAP
from src.scoring import ScoringCore


class ScoringStreamingService:
    def __init__(self, core: ScoringCore):
        self.core = core

        self.consumer = KafkaConsumer(
            'enriched-events',
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id='scoring-service',
            auto_offset_reset='earliest',
            value_deserializer=lambda v: json.loads(v.decode('utf-8'))
        )

        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            key_serializer=lambda k: k.encode('utf-8'),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def run(self):
        print("Scoring Service started. Consuming from 'enriched-events'...")
        count = 0

        while True:
            messages = self.consumer.poll(timeout_ms=1000)
            if not messages:
                continue

            for topic_partition, batch in messages.items():
                for msg in batch:
                    enriched = msg.value
                    scored = self.core.score_event(enriched)

                    self.producer.send(
                        'scored-events',
                        key=enriched['client_id'],
                        value=scored
                    )

                    count += 1
                    if count % 100 == 0:
                        self.producer.flush()
                        print(f"Scored {count} events")

    def close(self):
        self.consumer.close()
        self.producer.close()


if __name__ == '__main__':
    import redis
    from src.config import REDIS_HOST, REDIS_PORT
    from src.scoring.fusion import ScoreFusion

    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    fusion = ScoreFusion()
    core = ScoringCore(redis_client, fusion)
    service = ScoringStreamingService(core)

    try:
        service.run()
    except KeyboardInterrupt:
        print("\nScoring Service shutting down...")
    finally:
        service.close()
        redis_client.close()