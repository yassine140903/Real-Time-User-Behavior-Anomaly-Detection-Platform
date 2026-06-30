# scripts/test_e2e_pipeline.py

import json
import numpy as np
import redis
import psycopg2
from joblib import load as jl_load
from src.config import DB_CONFIG, REDIS_HOST, REDIS_PORT
from src.scoring.fusion import ScoreFusion
from src.scoring.ae_scorer import AEScorer
from src.scoring.lstm_scorer import LSTMScorer
from src.decision.decision import DecisionService, DecisionInput
from src.streaming.enrichment_service import EnrichmentService
from pathlib import Path

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

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    models_dir = Path("models")

    # ── Clear stale buffers from previous runs ──────────
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
    print(f"    Sample: client={events[0]['client_id'][:8]}... "
          f"op={events[0]['operation_type']} "
          f"amount={events[0]['amount']}")

    # ── Stage 1: Enrichment (warm-up + eval) ────────────
    print(f"\n[2] Running Enrichment (first 480 warm-up, last 20 eval)...")
    enrich = EnrichmentService.__new__(EnrichmentService)
    enrich.redis = r
    with open(models_dir / "feature_cols.json") as f:
        enrich.feature_cols = json.load(f)
    with open(models_dir / "lstm_config.json") as f:
        enrich.seq_len = json.load(f)["seq_len"]

    enriched_events = []
    for event in events:
        enriched = enrich.enrich_event(event)
        enriched_events.append(enriched)

    enriched_eval = enriched_events[-20:]
    print(f"    Warm-up: {len(enriched_events) - len(enriched_eval)} events")
    print(f"    Evaluating: {len(enriched_eval)} events")

    sample = enriched_eval[0]
    print(f"    z_amount={sample.get('z_amount', 'MISSING'):.4f}  "
          f"tx_count_24h={sample.get('tx_count_24h', 'MISSING')}  "
          f"tx_count_7d={sample.get('tx_count_7d', 'MISSING')}")

    # ── Feature diagnosis ───────────────────────────────
    fusion = ScoreFusion(str(models_dir))
    scaler = jl_load("models/scaler.pkl")
    sample_features = np.array([
        enriched_eval[0].get(col, 0.0) for col in fusion.feature_cols
    ])
    scaled = (sample_features - scaler.mean_) / scaler.scale_

    print("\n    Feature diagnosis (|z| > 3 = likely problem):")
    problem_count = 0
    for i, col in enumerate(fusion.feature_cols):
        if abs(scaled[i]) > 3:
            print(f"      {col}: raw={sample_features[i]:.4f}  "
                  f"train_mean={scaler.mean_[i]:.4f}  "
                  f"train_std={scaler.scale_[i]:.4f}  "
                  f"z={scaled[i]:.1f}")
            problem_count += 1
    if problem_count == 0:
        print("      None — all features within normal range")
    print(f"    Total problematic features: {problem_count}/30")

    # ── Stage 2: Scoring ────────────────────────────────
    print(f"\n[3] Running Scoring...")
    background = np.load(models_dir / "shap_background.npy")
    ae_scorer = AEScorer(
        model=fusion.ae, scaler=fusion.scaler,
        feature_cols=fusion.feature_cols, background_data=background
    )
    lstm_scorer = LSTMScorer(
        model=fusion.lstm, scaler=fusion.scaler,
        feature_cols=fusion.feature_cols, seq_len=fusion.seq_len
    )

    scored_events = []
    for enriched in enriched_eval:
        features = np.array([
            enriched.get(col, 0.0) for col in fusion.feature_cols
        ], dtype=np.float64)
        features_scaled = fusion.scaler.transform(features.reshape(1, -1))[0]

        seq_data = r.get(f"sequence:client:{enriched['client_id']}")
        sequence_scaled = None
        if seq_data:
            seq_raw = json.loads(seq_data)
            if len(seq_raw) >= fusion.seq_len:
                sequence_scaled = fusion.scaler.transform(
                    np.array(seq_raw[-fusion.seq_len:])
                )

        days = enriched.get('account_age_days', 0)
        result = fusion.score(features_scaled, sequence_scaled, days)

        ae_explanation = None
        lstm_explanation = None
        if result['fused_score'] >= 0.95:
            ae_explanation = ae_scorer.explain(features_scaled)
            if sequence_scaled is not None:
                lstm_explanation = lstm_scorer.explain(
                    sequence_scaled[-fusion.seq_len:], features_scaled
                )
        scored = {
            'event_id': enriched['event_id'],
            'client_id': enriched['client_id'],
            'account_id': enriched['account_id'],
            'employee_id': enriched['employee_id'],
            'branch_id': enriched['branch_id'],
            'timestamp': enriched['timestamp'],
            'amount': enriched['amount'],
            'operation_type': enriched['operation_type'],
            'z_amount': enriched.get('z_amount', 0.0),
            'tx_count_24h': enriched.get('tx_count_24h', 0),
            'cumulative_amount_24h': enriched.get('cumulative_amount_24h', 0.0),
            'near_threshold_count_7d': enriched.get('near_threshold_count_7d', 0),
            'same_employee_client_count_24h': enriched.get('same_employee_client_count_24h', 0),
            'is_new_counterparty': enriched.get('is_new_counterparty', 0),
            'is_round_amount': enriched.get('is_round_amount', 0),
            'has_duplicate_recent': enriched.get('has_duplicate_recent', 0),
            'account_age_days': enriched.get('account_age_days', 0),
            **result,
            'ae_explanation': ae_explanation,
            'lstm_explanation': lstm_explanation,
        }
        scored_events.append(scored)
    # Debug: check raw AE scores vs reference
    ae_raws = [s['ae_raw'] for s in scored_events]
    print(f"\n    AE raw scores: min={min(ae_raws):.6f}  max={max(ae_raws):.6f}")
    print(f"    AE ref distribution: min={fusion.ae_ref[0]:.6f}  "
          f"max={fusion.ae_ref[-1]:.6f}  "
          f"p95={fusion.ae_ref[int(len(fusion.ae_ref)*0.95)]:.6f}")
    scores = [s['fused_score'] for s in scored_events]
    print(f"    Score range: {min(scores):.4f} — {max(scores):.4f}")
    print(f"    Mean: {np.mean(scores):.4f}")
    high = [s for s in scored_events if s['fused_score'] >= 0.95]
    print(f"    Above REVIEW threshold: {len(high)}/{len(scored_events)}")
    if high:
        print(f"    SHAP explanation present: {high[0].get('ae_explanation') is not None}")

    # ── Stage 3: Decision ───────────────────────────────
    print(f"\n[4] Running Decision...")
    engine = DecisionService()
    decisions = []

    for scored in scored_events:
        cid = scored['client_id']
        profile_data = r.get(f"profile:client:{cid}")
        profile = json.loads(profile_data) if profile_data else None

        if profile:
            archetype = profile.get('archetype', 'unknown')
            reg_type = profile.get('client_type', 'habitual')
            maturity = profile.get('maturity_status', 'mature')
            home = profile.get('home_branch_id', '')
            daily_rate = sum(
                profile.get(f"tx_{op}_30d_count", 0) for op in OPERATIONS
            ) / 30.0
            daily_rate = max(daily_rate, 0.1)
        else:
            archetype, reg_type, maturity, home = 'unknown', 'habitual', 'mature', ''
            daily_rate = 1.0

        inp = DecisionInput(
            fused_score=scored['fused_score'],
            ae_pct=scored['ae_pct'],
            lstm_pct=scored.get('lstm_pct'),
            w_lstm=scored['w_lstm'],
            event_id=scored['event_id'],
            client_id=cid,
            employee_id=scored['employee_id'],
            branch_id=scored['branch_id'],
            operation_type=scored['operation_type'],
            amount=scored['amount'],
            timestamp=scored['timestamp'],
            z_amount=scored.get('z_amount', 0.0),
            tx_count_24h=int(scored.get('tx_count_24h', 0)),
            cumulative_amount_24h=scored.get('cumulative_amount_24h', 0.0),
            near_threshold_count_7d=int(scored.get('near_threshold_count_7d', 0)),
            same_employee_client_count_24h=int(scored.get('same_employee_client_count_24h', 0)),
            is_new_beneficiary=bool(scored.get('is_new_counterparty', 0)),
            is_round_amount=bool(scored.get('is_round_amount', 0)),
            has_duplicate_recent=bool(scored.get('has_duplicate_recent', 0)),
            days_since_opening=scored.get('account_age_days', 0),
            archetype=archetype,
            regulatory_client_type=reg_type,
            maturity_status=maturity,
            home_branch=home,
            expected_daily_rate=daily_rate,
            ae_explanation=scored.get('ae_explanation'),
            lstm_explanation=scored.get('lstm_explanation'),
        )

        decision = engine.decide(inp)
        decisions.append(decision)

    tier_counts = {}
    for d in decisions:
        tier_counts[d.tier.name] = tier_counts.get(d.tier.name, 0) + 1
    print(f"    Tier distribution: {tier_counts}")

    for d in decisions:
        if d.tier.value > 0:
            print(f"\n    ALERT: {d.event_id[:8]}...")
            print(f"      Tier: {d.tier.name}")
            print(f"      Score: {d.fused_score:.4f}")
            print(f"      Reasons: {d.reasons}")
            if d.regulatory_flags:
                print(f"      Reg flags: {d.regulatory_flags}")

    r.close()
    print(f"\n{'=' * 60}")
    print("E2E test complete")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    test_pipeline()