# scripts/debug_single_event.py

import json
import redis
import psycopg2
from pathlib import Path
from src.config import DB_CONFIG, REDIS_HOST, REDIS_PORT
from src.enrichment.core import EnrichmentCore
from src.scoring.core import ScoringCore
from src.decision.core import DecisionCore
from src.scoring.fusion import ScoreFusion
from src.decision.decision import DecisionService

EVENT_ID = "cbbd46c6-8dcf-4523-bef6-1e149a09a963"

def main():
    # Load the raw event
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT event_id, client_id, account_id, employee_id, branch_id,
               timestamp, amount, currency, channel, operation_type, payload
        FROM transactions WHERE event_id = %s
    """, (EVENT_ID,))
    columns = [desc[0] for desc in cursor.description]
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    event = dict(zip(columns, row))
    event['timestamp'] = event['timestamp'].isoformat()
    event['amount'] = float(event['amount'])

    print("=" * 60)
    print("RAW EVENT")
    print("=" * 60)
    print(f"  client_id: {event['client_id']}")
    print(f"  operation: {event['operation_type']}")
    print(f"  amount: {event['amount']}")
    print(f"  timestamp: {event['timestamp']}")

    # Wire up cores — same as streaming pipeline
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    models_dir = Path("models")

    enrichment_core = EnrichmentCore(redis_client=r, models_dir=str(models_dir))
    fusion = ScoreFusion(str(models_dir))
    scoring_core = ScoringCore(redis_client=r, fusion=fusion, models_dir=str(models_dir))
    engine = DecisionService()
    decision_core = DecisionCore(redis_client=r, engine=engine)

    # Stage 1: Enrichment
    print("\n" + "=" * 60)
    print("ENRICHMENT (direct)")
    print("=" * 60)
    enriched = enrichment_core.enrich_event(event)

    # Print all enriched features
    feature_cols = json.load(open("models/feature_cols.json"))
    for col in feature_cols:
        val = enriched.get(col, "MISSING")
        if isinstance(val, float):
            print(f"  {col}: {val:.6f}")
        else:
            print(f"  {col}: {val}")

    # Stage 2: Scoring
    print("\n" + "=" * 60)
    print("SCORING (direct)")
    print("=" * 60)
    scored = scoring_core.score_event(enriched)
    print(f"  ae_pct: {scored.get('ae_pct', 'MISSING')}")
    print(f"  lstm_pct: {scored.get('lstm_pct', 'MISSING')}")
    print(f"  w_lstm: {scored.get('w_lstm', 'MISSING')}")
    print(f"  fused_score: {scored['fused_score']:.6f}")

    # Stage 3: Decision
    print("\n" + "=" * 60)
    print("DECISION (direct)")
    print("=" * 60)
    decision = decision_core.decide_event(scored)
    print(f"  tier: {decision['tier']}")
    print(f"  fused_score: {decision['fused_score']:.6f}")
    print(f"  reasons: {decision['reasons']}")
    print(f"  regulatory_flags: {decision['regulatory_flags']}")

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  Streaming fused_score: 0.986742")
    print(f"  Direct fused_score:    {decision['fused_score']:.6f}")
    print(f"  Delta:                 {abs(0.986742 - decision['fused_score']):.6f}")

    r.close()

if __name__ == '__main__':
    main()