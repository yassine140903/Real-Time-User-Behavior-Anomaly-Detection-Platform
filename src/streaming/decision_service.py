# src/streaming/decision_service.py

import json
from kafka import KafkaConsumer, KafkaProducer
from src.config import KAFKA_BOOTSTRAP
from src.decision import DecisionCore
from src.streaming.dlq import ResilientConsumer
from src.streaming.logger import setup_logger


class DecisionStreamingService:
    def __init__(self, core: DecisionCore):
        self.core = core
        self.logger = setup_logger("decision")

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

        self.resilient = ResilientConsumer(
            service_name="decision",
            source_topic="scored-events",
            process_fn=self.core.decide_event,
            producer=self.producer,
            max_retries=3
        )

    def run(self):
        self.logger.info("Decision service started")
        count = 0
        alert_count = 0

        while True:
            messages = self.consumer.poll(timeout_ms=1000)
            if not messages:
                continue

            for topic_partition, batch in messages.items():
                for msg in batch:
                    scored = msg.value
                    output = self.resilient.handle(scored)

                    if output is not None:
                        self.producer.send(
                            'decisions',
                            key=scored['client_id'],
                            value=output
                        )
                        self.logger.info(
                            "Decision made",
                            extra={
                                "event_id": output.get("event_id"),
                                "client_id": output.get("client_id"),
                                "alert_tier": output.get("tier"),
                            },
                        )

                        if output["tier"] != "INFO":
                            alert_count += 1
                    else:
                        self.logger.warning(
                            "Event failed processing",
                            extra={"event_id": scored.get("event_id")},
                        )

                    count += 1
                    if count % 100 == 0:
                        self.producer.flush()
                        self.logger.info(f"Decided {count} events | {alert_count} alerts")

    def close(self):
        self.consumer.close()
        self.producer.close()


if __name__ == '__main__':
    import redis
    from src.config import REDIS_HOST, REDIS_PORT
    from src.decision.decision import DecisionService
    from src.monitoring.metrics import ServiceMetrics

    metrics = ServiceMetrics("decision")
    metrics.start_server(port=9102)

    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    engine = DecisionService()
    core = DecisionCore(redis_client, engine, metrics=metrics)
    service = DecisionStreamingService(core)

    try:
        service.run()
    except KeyboardInterrupt:
        print("\nDecision Service shutting down...")
    finally:
        service.close()
        redis_client.close()