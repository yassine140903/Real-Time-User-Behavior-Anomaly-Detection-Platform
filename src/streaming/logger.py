# src/streaming/logger.py

import json
import logging
import datetime


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""
    
    # Fields from the extra dict to include in output
    EXTRA_FIELDS = ("event_id", "client_id", "employee_id", "source_topic",
                    "error_type", "retry_count", "score", "alert_tier")
    
    def __init__(self, service_name):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "service": self.service_name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        
        # Pull in any extra fields that were passed
        for field in self.EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                log_entry[field] = value
        
        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)


def setup_logger(service_name, level=logging.INFO):

    logger = logging.getLogger(f"amen.{service_name}")
    logger.setLevel(level)
    
    # Avoid duplicate handlers if called twice
    if logger.handlers:
        return logger
    
    handler = logging.StreamHandler()  # stdout
    handler.setFormatter(JsonFormatter(service_name))
    logger.addHandler(handler)
    
    # Prevent propagation to root logger (no duplicate lines)
    logger.propagate = False
    
    return logger