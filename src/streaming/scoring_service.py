# src/streaming/scoring_service.py

import json
import numpy as np
import redis
from kafka import KafkaConsumer, KafkaProducer
from src.config import KAFKA_BOOTSTRAP, REDIS_HOST, REDIS_PORT
from src.scoring.fusion import ScoreFusion
from src.scoring.ae_scorer  import AEScorer
from src.scoring.lstm_scorer import LSTMScorer
from pathlib import Path


class ScoringService:
    # Only explain when score crosses REVIEW threshold
    EXPLAIN_THRESHOLD = 0.95

    def __init__(self, models_dir="models"):
        models_dir = Path(models_dir)

        # ── Kafka ───────────────────────────────────────────
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

        # ── Redis (for sequence buffer) ─────────────────────
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

        # ── Models ──────────────────────────────────────────
        self.fusion = ScoreFusion(str(models_dir))

        # Scorers share model instances from fusion (no double-loading)
        background = np.load(models_dir / "shap_background.npy")
        self.ae_scorer = AEScorer(
            model=self.fusion.ae,
            scaler=self.fusion.scaler,
            feature_cols=self.fusion.feature_cols,
            background_data=background
        )
        self.lstm_scorer = LSTMScorer(
            model=self.fusion.lstm,
            scaler=self.fusion.scaler,
            feature_cols=self.fusion.feature_cols,
            seq_len=self.fusion.seq_len
        )

    def get_sequence(self, client_id):
        """Pull LSTM sequence buffer from Redis."""
        data = self.redis.get(f"sequence:client:{client_id}")
        if data:
            return json.loads(data)
        return None

    def score_event(self, enriched):
        """Score a single enriched event."""
        cid = enriched['client_id']

        # ── Extract feature vector in correct order ─────────
        features = np.array([
            enriched.get(col, 0.0) for col in self.fusion.feature_cols
        ], dtype=np.float64)

        # ── Scale current event ─────────────────────────────
        features_scaled = self.fusion.scaler.transform(
            features.reshape(1, -1)
        )[0]

        # ── Pull and scale sequence ─────────────────────────
        seq_raw = self.get_sequence(cid)
        sequence_scaled = None
        if seq_raw and len(seq_raw) >= self.fusion.seq_len:
            sequence_scaled = self.fusion.scaler.transform(
                np.array(seq_raw[-self.fusion.seq_len:])
            )

        # ── Fuse scores ────────────────────────────────────
        days = enriched.get('account_age_days', 0)
        result = self.fusion.score(features_scaled, sequence_scaled, days)

        # ── Conditional SHAP explanation ────────────────────
        ae_explanation = None
        lstm_explanation = None

        if result['fused_score'] >= self.EXPLAIN_THRESHOLD:
            ae_explanation = self.ae_scorer.explain(features_scaled)

            if sequence_scaled is not None:
                lstm_explanation = self.lstm_scorer.explain(
                    sequence_scaled[-self.fusion.seq_len:],
                    features_scaled
                )

        # ── Build output ────────────────────────────────────
        output = {
            # Pass-through identifiers
            'event_id': enriched['event_id'],
            'client_id': cid,
            'account_id': enriched['account_id'],
            'employee_id': enriched['employee_id'],
            'branch_id': enriched['branch_id'],
            'timestamp': enriched['timestamp'],
            'amount': enriched['amount'],
            'operation_type': enriched['operation_type'],

            # Pass-through enriched features (Decision Service needs these)
            'z_amount': enriched.get('z_amount', 0.0),
            'tx_count_24h': enriched.get('tx_count_24h', 0),
            'cumulative_amount_24h': enriched.get('cumulative_amount_24h', 0.0),
            'near_threshold_count_7d': enriched.get('near_threshold_count_7d', 0),
            'same_employee_client_count_24h': enriched.get('same_employee_client_count_24h', 0),
            'is_new_counterparty': enriched.get('is_new_counterparty', 0),
            'is_round_amount': enriched.get('is_round_amount', 0),
            'has_duplicate_recent': enriched.get('has_duplicate_recent', 0),
            'account_age_days': enriched.get('account_age_days', 0),

            # Scores
            'ae_raw': result['ae_raw'],
            'lstm_raw': result['lstm_raw'],
            'ae_pct': result['ae_pct'],
            'lstm_pct': result['lstm_pct'],
            'w_lstm': result['w_lstm'],
            'fused_score': result['fused_score'],

            # Explanations (None for INFO events)
            'ae_explanation': ae_explanation,
            'lstm_explanation': lstm_explanation,
        }
        return output

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
                    scored = self.score_event(enriched)

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
        self.redis.close()


if __name__ == '__main__':
    service = ScoringService()
    try:
        service.run()
    except KeyboardInterrupt:
        print("\nScoring Service shutting down...")
    finally:
        service.close()