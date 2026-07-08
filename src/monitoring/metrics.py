# src/monitoring/metrics.py

from prometheus_client import Counter, Histogram, start_http_server


class ServiceMetrics:
    def __init__(self, service_name: str):
        self.service_name = service_name

        # --- Shared across all services ---
        self.events_processed = Counter(
            f"{service_name}_events_processed_total",
            f"Total events processed by {service_name}",
            ["operation_type"]
        )

        self.errors = Counter(
            f"{service_name}_errors_total",
            f"Total processing errors in {service_name}",
            ["error_class"]
        )

        self.processing_duration = Histogram(
            f"{service_name}_processing_duration_seconds",
            f"Event processing latency in {service_name}",
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
        )

        # --- Scoring-specific ---
        self.fused_score = Histogram(
            f"{service_name}_fused_score_distribution",
            "Distribution of fused anomaly scores",
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0]
        )

    def start_server(self, port: int):
        start_http_server(port)