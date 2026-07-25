# src/streaming/dlq.py

import time
import traceback
from datetime import datetime, timezone

import psycopg2
import redis

from src.streaming.logger import setup_logger

logger = setup_logger("dlq")

TRANSIENT_ERRORS = (
    redis.ConnectionError,
    redis.TimeoutError,
    redis.BusyLoadingError,
    psycopg2.OperationalError,
    psycopg2.InterfaceError,
    ConnectionError,
    TimeoutError,
    OSError,
)

DLQ_TOPIC = "dlq"


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def build_dlq_envelope(original_event, source_topic, service_name, error_type,
                        error_message, retry_count, max_retries):
    now = _now_iso()
    return {
        "original_event": original_event,
        "source_topic": source_topic,
        "service_name": service_name,
        "error_type": error_type,
        "error_message": error_message,
        "retry_count": retry_count,
        "max_retries": max_retries,
        "stacktrace": traceback.format_exc(),
        "first_failure_at": now,
        "last_failure_at": now,
    }


class ResilientConsumer:
    def __init__(self, service_name, source_topic, process_fn, producer,
                 max_retries=3, base_delay=1.0):
        self.service_name = service_name
        self.source_topic = source_topic
        self.process_fn = process_fn
        self.producer = producer
        self.max_retries = max_retries
        self.base_delay = base_delay

    def handle(self, event):
        event_id = event.get("event_id") if isinstance(event, dict) else None
        client_id = event.get("client_id") if isinstance(event, dict) else None
        try:
            return self.process_fn(event)
        except TRANSIENT_ERRORS as e:
            return self._retry_with_backoff(event, e)
        except Exception as e:
            logger.warning(
                "Poison pill detected",
                extra={"event_id": event_id, "client_id": client_id, "error_type": "poison_pill"},
            )
            self._send_to_dlq(event, type(e).__name__, e, retry_count=0,
                               max_retries=self.max_retries)
            return None

    def _retry_with_backoff(self, event, initial_error):
        event_id = event.get("event_id") if isinstance(event, dict) else None
        last_error = initial_error
        for attempt in range(1, self.max_retries + 1):
            logger.info(
                "Transient error, retrying",
                extra={"event_id": event_id, "retry_count": attempt, "error_type": "transient"},
            )
            delay = self.base_delay * (2 ** (attempt - 1))
            time.sleep(delay)
            try:
                return self.process_fn(event)
            except TRANSIENT_ERRORS as e:
                last_error = e
                continue
            except Exception as e:
                logger.warning(
                    "Poison pill detected",
                    extra={"event_id": event_id, "client_id": event.get("client_id") if isinstance(event, dict) else None,
                           "error_type": "poison_pill"},
                )
                self._send_to_dlq(event, type(e).__name__, e, retry_count=attempt,
                                   max_retries=self.max_retries)
                return None

        logger.error(
            "Retries exhausted, sending to DLQ",
            extra={"event_id": event_id, "retry_count": self.max_retries},
        )
        self._send_to_dlq(event, type(last_error).__name__, last_error,
                           retry_count=self.max_retries, max_retries=self.max_retries)
        return None

    def _send_to_dlq(self, event, error_type, error, retry_count, max_retries):
        event_id = event.get("event_id") if isinstance(event, dict) else None
        envelope = build_dlq_envelope(
            original_event=event,
            source_topic=self.source_topic,
            service_name=self.service_name,
            error_type=error_type,
            error_message=str(error),
            retry_count=retry_count,
            max_retries=max_retries,
        )
        try:
            key = event.get("client_id") if isinstance(event, dict) else None
            if key:
                self.producer.send(DLQ_TOPIC, key=key, value=envelope)
            else:
                self.producer.send(DLQ_TOPIC, value=envelope)
            logger.info(
                "Event sent to DLQ",
                extra={"event_id": event_id, "error_type": error_type},
            )
        except Exception:
            logger.critical(
                "Failed to send to DLQ",
                extra={"event_id": event_id},
                exc_info=True,
            )
            logger.critical(
                "Lost event",
                extra={"event_id": event_id},
            )
