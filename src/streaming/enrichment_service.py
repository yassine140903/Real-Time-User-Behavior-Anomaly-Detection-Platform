# src/streaming/enrichment_service.py

import json
from kafka import KafkaConsumer, KafkaProducer
from src.config import KAFKA_BOOTSTRAP
from src.enrichment import EnrichmentCore
from src.streaming.dlq import ResilientConsumer
from src.streaming.logger import setup_logger


class EnrichmentStreamingService:
    def __init__(self, core: EnrichmentCore):
        self.core = core
        self.logger = setup_logger("enrichment")

        self.consumer = KafkaConsumer(
            'raw-events',
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id='enrichment-service',
            auto_offset_reset='earliest',
            value_deserializer=lambda v: json.loads(v.decode('utf-8'))
        )

        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            key_serializer=lambda k: k.encode('utf-8'),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        self.resilient = ResilientConsumer(
            service_name="enrichment",
            source_topic="raw-events",
            process_fn=self.core.enrich_event,
            producer=self.producer,
            max_retries=3
        )

    def run(self):
        self.logger.info("Enrichment service started")
        count = 0

        while True:
            messages = self.consumer.poll(timeout_ms=1000)
            if not messages:
                continue

            for topic_partition, batch in messages.items():
                for msg in batch:
                    event = msg.value
                    result = self.resilient.handle(event)

                    if result is not None:
                        self.producer.send(
                            'enriched-events',
                            key=event['client_id'],
                            value=result
                        )
                        self.logger.info(
                            "Event enriched",
                            extra={"event_id": event.get("event_id"), "client_id": event.get("client_id")},
                        )
                    else:
                        self.logger.warning(
                            "Event failed processing",
                            extra={"event_id": event.get("event_id")},
                        )

                    count += 1
                    if count % 100 == 0:
                        self.producer.flush()
                        self.logger.info(f"Enriched {count} events")

    def close(self):
        self.consumer.close()
        self.producer.close()


if __name__ == '__main__':
    import redis
    from src.config import REDIS_HOST, REDIS_PORT
    from src.monitoring.metrics import ServiceMetrics

    metrics = ServiceMetrics("enrichment")
    metrics.start_server(port=9100)

    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    core = EnrichmentCore(redis_client, metrics=metrics)
    service = EnrichmentStreamingService(core)

    try:
        service.run()
    except KeyboardInterrupt:
        print("\nEnrichment Service shutting down...")
    finally:
        service.close()
        redis_client.close()