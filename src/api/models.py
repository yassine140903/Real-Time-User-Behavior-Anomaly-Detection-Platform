# src/api/models.py

from typing import Optional, Union
from pydantic import BaseModel, model_validator
from enum import Enum
from typing import Literal

class OperationType(str, Enum):
    retrait = "retrait"
    versement = "versement"
    virement = "virement"
    cheque = "cheque"


# ── PAYLOAD MODELS ──────────────────────────────────────────

class VirementPayload(BaseModel):
    beneficiary_id: str

class ChequePayload(BaseModel):
    emitter_id: str


# ── REQUEST MODEL ───────────────────────────────────────────

class SimulationRequest(BaseModel):
    client_id: str
    employee_id: str
    amount: float
    operation_type: OperationType
    payload: Optional[Union[VirementPayload, ChequePayload]] = None

    @model_validator(mode="after")
    def validate_payload_matches_operation(self):
        op = self.operation_type

        if op == OperationType.virement:
            if not isinstance(self.payload, VirementPayload):
                raise ValueError("virement requires payload with beneficiary_id")

        elif op == OperationType.cheque:
            if not isinstance(self.payload, ChequePayload):
                raise ValueError("cheque requires payload with emitter_id")

        else:
            if self.payload is not None:
                raise ValueError(f"{op.value} does not accept a payload")

        return self


# ── RESPONSE MODEL ──────────────────────────────────────────

class SimulationResponse(BaseModel):
    event_id: str
    client_id: str
    account_id: str
    employee_id: str
    branch_id: str
    timestamp: str
    amount: float
    operation_type: str

    # Decision
    tier: str
    fused_score: float
    reasons: list[str]
    regulatory_flags: list[str]

    # Explainability
    ae_explanation: Optional[dict] = None
    lstm_explanation: Optional[dict] = None


class FeedbackRequest(BaseModel):
    event_id: str  # UUID as string, matches event_id in alerts
    decision: Literal["confirmed", "rejected"]
    comment: Optional[str] = None