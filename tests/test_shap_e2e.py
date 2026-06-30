"""
End-to-end test: score → explain → decide.
tests/test_shap_e2e.py
"""

import numpy as np
import pandas as pd
import json

from src.scoring.fusion import ScoreFusion
from src.scoring.AEScorer import AEScorer
from src.scoring.lstm_scorer import LSTMScorer
from src.decision.decision import DecisionService, DecisionInput


def main():
    # ── 1. Load fusion (single model load) ───────────────
    fusion = ScoreFusion()

    # ── 2. Load data ─────────────────────────────────────
    df = pd.read_csv("data/enriched_events.csv", low_memory=False)
    raw_df = pd.read_csv("data/transactions.csv")

    # ── 3. Build background data ─────────────────────────
    normal = df[df["is_anomaly"] == False].sample(100, random_state=42)
    background = fusion.scaler.transform(normal[fusion.feature_cols].values)
    np.save("models/shap_background.npy", background)

    # ── 4. Init scorers (share models from fusion) ───────
    ae_scorer = AEScorer(
        model=fusion.ae,
        scaler=fusion.scaler,
        feature_cols=fusion.feature_cols,
        background_data=background
    )

    lstm_scorer = LSTMScorer(
        model=fusion.lstm,
        scaler=fusion.scaler,
        feature_cols=fusion.feature_cols,
        seq_len=fusion.seq_len
    )

    # ── 5. Pick an anomalous event ───────────────────────
# ── 5. Pick an anomalous event with enough history ───
    anomalous_events = df[df["is_anomaly"] == True]
    
    chosen = None
    for _, row in anomalous_events.iterrows():
        cid = row["client_id"]
        normal_count = len(df[(df["client_id"] == cid) & 
                              (df["is_anomaly"] == False)])
        if normal_count >= fusion.seq_len:
            chosen = row
            break
    
    if chosen is None:
        print("No anomalous event with enough client history found.")
        return
    
    anomalous = chosen
    event_id = anomalous["event_id"]
    client_id = anomalous["client_id"]
    print(f"Selected event {event_id}")
    print(f"  Client: {client_id}")
    print(f"  Anomaly type: {anomalous.get('anomaly_type', 'unknown')}")

    # Get raw event fields (operation_type, employee_id, branch_id)
    raw_event = raw_df[raw_df["event_id"] == event_id].iloc[0]

    # Scale enriched features
    x_raw = anomalous[fusion.feature_cols].values.astype(np.float32)
    x_scaled = fusion.scaler.transform(x_raw.reshape(1, -1))[0]

    # ── 6. AE: score + explain ───────────────────────────
    ae_result = ae_scorer.explain(x_scaled, top_k=3)
    print("\n── AE Explanation ──")
    print(f"  Score: {ae_result['score']}")
    for feat, (shap_val, raw_val) in ae_result["top_features"].items():
        print(f"  {feat}: SHAP={shap_val}, value={raw_val}")

    # ── 7. LSTM: score + explain ─────────────────────────
    client_events = df[(df["client_id"] == client_id) &
                       (df["is_anomaly"] == False)].tail(fusion.seq_len)

    lstm_result = None
    seq_scaled = None
    if len(client_events) == fusion.seq_len:
        seq_scaled = fusion.scaler.transform(
            client_events[fusion.feature_cols].values)
        lstm_result = lstm_scorer.explain(seq_scaled, x_scaled, top_k=3)
        print("\n── LSTM Explanation ──")
        print(f"  Score: {lstm_result['score']}")
        for feat, (expected, actual) in lstm_result["top_deviations"].items():
            print(f"  {feat}: expected={expected}, actual={actual}")
    else:
        print("\n── LSTM: not enough sequence data ──")

    # ── 8. Fuse scores ───────────────────────────────────
    days = int(anomalous.get("account_age_days", 180))
    fused = fusion.score(x_scaled,
                         sequence_scaled=seq_scaled if lstm_result else None,
                         days_since_opening=days)

    # ── 9. Decision ──────────────────────────────────────
    svc = DecisionService()
    inp = DecisionInput(
        fused_score=fused["fused_score"],
        ae_pct=fused["ae_pct"],
        lstm_pct=fused["lstm_pct"],
        w_lstm=fused["w_lstm"],
        event_id=str(event_id),
        client_id=str(client_id),
        employee_id=str(raw_event["employee_id"]),
        branch_id=str(raw_event["branch_id"]),
        operation_type=str(raw_event["operation_type"]),
        amount=float(raw_event["amount"]),
        timestamp=str(anomalous["timestamp"]),
        z_amount=float(anomalous.get("z_amount", 0)),
        tx_count_24h=int(anomalous.get("tx_count_24h", 0)),
        cumulative_amount_24h=float(anomalous.get("cumulative_amount_24h", 0)),
        near_threshold_count_7d=int(anomalous.get("near_threshold_count_7d", 0)),
        is_new_beneficiary=bool(anomalous.get("is_new_counterparty", False)),
        is_round_amount=bool(anomalous.get("is_round_amount", False)),
        has_duplicate_recent=bool(anomalous.get("has_duplicate_recent", False)),
        days_since_opening=days,
        ae_explanation=ae_result,
        lstm_explanation=lstm_result,
    )

    decision = svc.decide(inp)

    # ── 10. Final output ─────────────────────────────────
    print("\n── Decision ──")
    print(f"  Tier: {decision.tier.name}")
    print(f"  Fused score: {decision.fused_score}")

    if decision.explanation:
        print(f"  Trigger: {decision.explanation['trigger']}")
        print(f"  Explanation for supervisor:")
        for r in decision.explanation["reasons"]:
            print(f"    • {r}")
    else:
        print("  No explanation (INFO tier)")


if __name__ == "__main__":
    main()