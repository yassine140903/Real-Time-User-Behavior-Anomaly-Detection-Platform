# src/streaming/scoring_service.py

import json
from kafka import KafkaConsumer, KafkaProducer
from src.config import KAFKA_BOOTSTRAP
from src.scoring import ScoringCore
from src.streaming.dlq import ResilientConsumer
from src.streaming.logger import setup_logger


class ScoringStreamingService:
    def __init__(self, core: ScoringCore):
        self.core = core
        self.logger = setup_logger("scoring")

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

        self.resilient = ResilientConsumer(
            service_name="scoring",
            source_topic="enriched-events",
            process_fn=self.core.score_event,
            producer=self.producer,
            max_retries=3
        )

    def run(self):
        self.logger.info("Scoring service started")
        count = 0

        while True:
            messages = self.consumer.poll(timeout_ms=1000)
            if not messages:
                continue

            for topic_partition, batch in messages.items():
                for msg in batch:
                    enriched = msg.value
                    result = self.resilient.handle(enriched)

                    if result is not None:
                        self.producer.send(
                            'scored-events',
                            key=enriched['client_id'],
                            value=result
                        )
                        self.logger.info(
                            "Event scored",
                            extra={
                                "event_id": enriched.get("event_id"),
                                "client_id": enriched.get("client_id"),
                                "score": result.get("fused_score") if isinstance(result, dict) else None,
                            },
                        )
                    else:
                        self.logger.warning(
                            "Event failed processing",
                            extra={"event_id": enriched.get("event_id")},
                        )

                    count += 1
                    if count % 100 == 0:
                        self.producer.flush()
                        self.logger.info(f"Scored {count} events")

    def close(self):
        self.consumer.close()
        self.producer.close()


if __name__ == '__main__':
    import redis
    from src.config import REDIS_HOST, REDIS_PORT
    from src.scoring.fusion import ScoreFusion
    from src.monitoring.metrics import ServiceMetrics

    metrics = ServiceMetrics("scoring")
    metrics.start_server(port=9101)

    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    fusion = ScoreFusion()
    core = ScoringCore(redis_client, fusion, metrics=metrics)
    service = ScoringStreamingService(core)

    try:
        service.run()
    except KeyboardInterrupt:
        print("\nScoring Service shutting down...")
    finally:
        service.close()
        redis_client.close()