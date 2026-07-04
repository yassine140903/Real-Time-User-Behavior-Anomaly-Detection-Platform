# scripts/test_e2e_pipeline.py

import json
import numpy as np
import redis
import psycopg2
from pathlib import Path
from src.config import DB_CONFIG, REDIS_HOST, REDIS_PORT
from src.enrichment.core import EnrichmentCore
from src.scoring.core import ScoringCore
from src.decision.core import DecisionCore
from src.scoring.fusion import ScoreFusion
from src.decision.decision import DecisionService

OPERATIONS = ["retrait", "versement", "virement", "cheque"]


def load_sample_events(n=500):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT event_id, client_id, account_id, employee_id, branch_id,
            timestamp, amount, currency, channel, operation_type, payload
        FROM transactions
        ORDER BY timestamp
        LIMIT %s
    """, (n,))
    columns = [desc[0] for desc in cursor.description]
    events = []
    for row in cursor.fetchall():
        event = dict(zip(columns, row))
        event['timestamp'] = event['timestamp'].isoformat()
        event['amount'] = float(event['amount'])
        events.append(event)
    cursor.close()
    conn.close()
    return events


def test_pipeline():
    print("=" * 60)
    print("E2E Pipeline Test (direct, no Kafka)")
    print("=" * 60)

    # ── Wiring (composition root) ───────────────────────
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    models_dir = Path("models")

    enrichment_core = EnrichmentCore(redis_client=r, models_dir=str(models_dir))
    fusion = ScoreFusion(str(models_dir))
    scoring_core = ScoringCore(redis_client=r, fusion=fusion, models_dir=str(models_dir))
    engine = DecisionService()
    decision_core = DecisionCore(redis_client=r, engine=engine)

    # ── Clear stale buffers ─────────────────────────────
    for key in r.keys("buffer:client:*"):
        r.delete(key)
    for key in r.keys("buffer:employee:*"):
        r.delete(key)
    for key in r.keys("sequence:client:*"):
        r.delete(key)
    print("Cleared stale buffers")

    # ── Load sample events ──────────────────────────────
    events = load_sample_events(500)
    print(f"\n[1] Loaded {len(events)} raw events from PostgreSQL")

    # ── Stage 1: Enrichment (warm-up + eval) ────────────
    print(f"\n[2] Running Enrichment (first 480 warm-up, last 20 eval)...")
    enriched_events = []
    for event in events:
        enriched = enrichment_core.enrich_event(event)
        enriched_events.append(enriched)

    enriched_eval = enriched_events[-20:]
    print(f"    Warm-up: {len(enriched_events) - len(enriched_eval)} events")
    print(f"    Evaluating: {len(enriched_eval)} events")

    sample = enriched_eval[0]
    print(f"    z_amount={sample.get('z_amount', 'MISSING'):.4f}  "
          f"tx_count_24h={sample.get('tx_count_24h', 'MISSING')}  "
          f"tx_count_7d={sample.get('tx_count_7d', 'MISSING')}")

    # ── Stage 2: Scoring ────────────────────────────────
    print(f"\n[3] Running Scoring...")
    scored_events = []
    for enriched in enriched_eval:
        scored = scoring_core.score_event(enriched)
        scored_events.append(scored)

    scores = [s['fused_score'] for s in scored_events]
    print(f"    Score range: {min(scores):.4f} — {max(scores):.4f}")
    print(f"    Mean: {np.mean(scores):.4f}")
    high = [s for s in scored_events if s['fused_score'] >= 0.95]
    print(f"    Above REVIEW threshold: {len(high)}/{len(scored_events)}")
    if high:
        print(f"    SHAP explanation present: {high[0].get('ae_explanation') is not None}")

    # ── Stage 3: Decision ───────────────────────────────
    print(f"\n[4] Running Decision...")
    decisions = []
    for scored in scored_events:
        decision = decision_core.decide_event(scored)
        decisions.append(decision)

    tier_counts = {}
    for d in decisions:
        tier_counts[d["tier"]] = tier_counts.get(d["tier"], 0) + 1
    print(f"    Tier distribution: {tier_counts}")

    for d in decisions:
        if d["tier"] != "INFO":
            print(f"\n    ALERT: {d['event_id'][:8]}...")
            print(f"      Tier: {d['tier']}")
            print(f"      Score: {d['fused_score']:.4f}")
            print(f"      Reasons: {d['reasons']}")
            if d["regulatory_flags"]:
                print(f"      Reg flags: {d['regulatory_flags']}")
            if d.get("ae_explanation"):
                print(f"      AE explanation: {json.dumps(d['ae_explanation'])[:80]}...")
            if d.get("lstm_explanation"):
                print(f"      LSTM explanation: {json.dumps(d['lstm_explanation'])[:80]}...")

    r.close()
    print(f"\n{'=' * 60}")
    print("E2E test complete")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    test_pipeline()