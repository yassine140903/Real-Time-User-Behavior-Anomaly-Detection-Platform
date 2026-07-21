# src/api/routes.py

import uuid
import json
from datetime import datetime
from fastapi import APIRouter, Request
from src.api.models import SimulationRequest, SimulationResponse, OperationType, FeedbackRequest
from fastapi import APIRouter, Request, HTTPException
from datetime import datetime, timezone



router = APIRouter()


def _build_raw_event(req: SimulationRequest, account_id: str) -> dict:
    """Transform a simulation request into a raw event dict."""

    # Build payload based on operation type
    if req.operation_type == OperationType.retrait:
        payload = {"mode": "especes"}
    elif req.operation_type == OperationType.versement:
        payload = {"depositor_id": None}
    elif req.operation_type == OperationType.virement:
        payload = {"beneficiary_id": req.payload.beneficiary_id, "motif": "simulation"}
    elif req.operation_type == OperationType.cheque:
        payload = {"emitter_id": req.payload.emitter_id, "cheque_number": f"SIM-{uuid.uuid4().hex[:8]}"}

    return {
        "event_id": str(uuid.uuid4()),
        "client_id": req.client_id,
        "account_id": account_id,
        "employee_id": req.employee_id,
        "branch_id": "SIM-000",
        "timestamp": datetime.now().isoformat(),
        "amount": req.amount,
        "currency": "TND",
        "channel": "guichet",
        "operation_type": req.operation_type.value,
        "payload": json.dumps(payload),
    }


@router.post("/score", response_model=SimulationResponse)
def simulate_score(req: SimulationRequest, request: Request):
    state = request.app.state

    raw_event = _build_raw_event(req, account_id="N/A")

    enriched = state.enrichment_core.enrich_event(raw_event, dry_run=True)
    scored = state.scoring_core.score_event(enriched)
    decision = state.decision_core.decide_event(scored)

    return decision

@router.post("/feedback")
def submit_feedback(req: FeedbackRequest, request: Request):
    conn = request.app.state.pg_pool.getconn()
    try:
        with conn.cursor() as cur:
            # Update the alert
            cur.execute("""
                UPDATE alerts 
                SET supervisor_decision = %s,
                    supervisor_notes = %s,
                    decision_timestamp = %s
                WHERE event_id = %s
            """, (req.decision, req.comment, datetime.now(timezone.utc), req.alert_id))
            
            if cur.rowcount == 0:
                conn.rollback()
                raise HTTPException(status_code=404, detail="Alert not found")
            
            # Insert into evaluation_labels for retraining feedback
            cur.execute("""
                INSERT INTO evaluation_labels (event_id, is_anomaly, anomaly_type)
                VALUES (%s, %s, %s)
                ON CONFLICT (event_id) DO UPDATE
                SET is_anomaly = EXCLUDED.is_anomaly,
                    anomaly_type = EXCLUDED.anomaly_type
            """, (
                req.alert_id,
                req.decision == "confirmed",
                "supervisor_confirmed" if req.decision == "confirmed" else None
            ))
            
            conn.commit()
        return {"status": "ok", "event_id": req.alert_id, "decision": req.decision}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        request.app.state.pg_pool.putconn(conn)