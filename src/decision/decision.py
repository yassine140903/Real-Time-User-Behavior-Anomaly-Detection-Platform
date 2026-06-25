# src/decision/decision.py
"""
Decision Service.

Consumes scored events (fused score + raw event + profile data)
and assigns alert tiers: INFO / REVIEW / ALERT / BLOCK.

Four layers of logic applied in order:
  1. Score-based tier assignment (from calibrated thresholds)
  2. Regulatory rule escalation (hard LAB/FT overrides)
  3. Risk history adjustment (recidivist escalation / false-positive dampening)
  4. Explanation assembly (human-readable reasons based on trigger type)

Tiers can only be ESCALATED by rules, never de-escalated.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from enum import IntEnum
from src.decision.explainer import assemble_explanation


class AlertTier(IntEnum):
    """Ordered so comparison works: BLOCK > ALERT > REVIEW > INFO."""
    INFO   = 0
    REVIEW = 1
    ALERT  = 2
    BLOCK  = 3


@dataclass
class DecisionInput:
    """Everything the Decision Service needs to make a ruling."""
    # ── From Scoring Service ────────────────────────────
    fused_score: float
    ae_pct: float
    lstm_pct: Optional[float]
    w_lstm: float

    # ── From raw event ──────────────────────────────────
    event_id: str
    client_id: str
    employee_id: str
    branch_id: str
    operation_type: str
    amount: float
    timestamp: str

    # ── From enriched features ──────────────────────────
    z_amount: float = 0.0
    tx_count_24h: int = 0
    cumulative_amount_24h: float = 0.0
    near_threshold_count_7d: int = 0
    same_employee_client_count_24h: int = 0
    is_new_beneficiary: bool = False
    is_round_amount: bool = False
    has_duplicate_recent: bool = False

    # ── From client profile (stubbed until streaming) ───
    days_since_opening: int = 0
    archetype: str = "unknown"
    regulatory_client_type: str = "habitual"
    maturity_status: str = "mature"
    home_branch: str = ""

    # Risk history (stubbed — no alerts table yet)
    alert_count_30d: int = 0
    confirmed_count_30d: int = 0
    rejected_count_30d: int = 0
    days_since_last_alert: int = 999
    expected_daily_rate: float = 1.0

    # ── From Scoring Service (explanations) ─────────────
    ae_explanation: Optional[dict] = None
    lstm_explanation: Optional[dict] = None


@dataclass
class Decision:
    """Output of the Decision Service."""
    event_id: str
    tier: AlertTier
    fused_score: float
    reasons: List[str] = field(default_factory=list)
    regulatory_flags: List[str] = field(default_factory=list)
    explanation: Optional[dict] = None

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "tier": self.tier.name,
            "fused_score": round(self.fused_score, 6),
            "reasons": self.reasons,
            "regulatory_flags": self.regulatory_flags,
            "explanation": self.explanation,
        }


class DecisionService:
    """
    Stateless decision engine.

    All state (profile, risk history) arrives in DecisionInput.
    This service applies rules and returns a Decision.
    """

    # ── Score-based thresholds (calibrated on validation set) ──
    T1 = 0.95
    T2 = 0.99
    T3 = 0.999

    # ── Regulatory constants ───────────────────────────────────
    REPORTING_THRESHOLD = 10_000
    SMURFING_BAND_LOW  = 8_000
    SMURFING_BAND_HIGH = 9_999
    CHEQUE_CEILING     = 30_000
    PASSEUR_VERSEMENT_LIMIT = 5
    ROUND_AMOUNT_VALUES = {1000, 2000, 3000, 5000, 10000, 15000, 20000}

    # ── Risk history thresholds ────────────────────────────────
    RECIDIVIST_CONFIRMED_30D = 3
    FALSE_POSITIVE_REJECTED_30D = 3

    def decide(self, inp: DecisionInput) -> Decision:
        """
        Main entry point. Returns a Decision with tier + reasons + explanation.
        """
        reasons = []
        reg_flags = []

        # ════════════════════════════════════════════════════════
        # LAYER 1: Score-based tier
        # ════════════════════════════════════════════════════════
        if inp.fused_score >= self.T3:
            tier = AlertTier.BLOCK
            reasons.append(f"Fused score {inp.fused_score:.4f} >= BLOCK threshold")
        elif inp.fused_score >= self.T2:
            tier = AlertTier.ALERT
            reasons.append(f"Fused score {inp.fused_score:.4f} >= ALERT threshold")
        elif inp.fused_score >= self.T1:
            tier = AlertTier.REVIEW
            reasons.append(f"Fused score {inp.fused_score:.4f} >= REVIEW threshold")
        else:
            tier = AlertTier.INFO
            reasons.append(f"Fused score {inp.fused_score:.4f} below all thresholds")

        # ════════════════════════════════════════════════════════
        # LAYER 2: Regulatory rule escalation
        # ════════════════════════════════════════════════════════

        # Rule 1: Mandatory reporting threshold (context-aware)
        if inp.amount >= self.REPORTING_THRESHOLD:
            reg_flags.append(
                f"Amount {inp.amount:.2f} DT >= {self.REPORTING_THRESHOLD} DT "
                f"reporting threshold (BCT)")
            if inp.z_amount >= 2.0:
                tier = max(tier, AlertTier.ALERT)

        # Rule 2: Smurfing band detection
        if (self.SMURFING_BAND_LOW <= inp.amount <= self.SMURFING_BAND_HIGH
                and inp.near_threshold_count_7d >= 2):
            reg_flags.append(
                f"Amount {inp.amount:.2f} DT in smurfing band "
                f"({self.SMURFING_BAND_LOW}-{self.SMURFING_BAND_HIGH} DT) "
                f"with {inp.near_threshold_count_7d} near-threshold txns in 7d")
            tier = max(tier, AlertTier.ALERT)

        # Rule 3: Cheque structuring (Law n°41-2024)
        if (inp.operation_type == "cheque"
                and 25_000 <= inp.amount < self.CHEQUE_CEILING):
            reg_flags.append(
                f"Cheque amount {inp.amount:.2f} DT near "
                f"{self.CHEQUE_CEILING} DT ceiling (Law n°41-2024)")
            tier = max(tier, AlertTier.REVIEW)

        # Rule 4: Passeur de fonds (CTAF September 2021)
        if (inp.regulatory_client_type == "occasional"
                and inp.operation_type == "versement"
                and inp.tx_count_24h >= self.PASSEUR_VERSEMENT_LIMIT):
            reg_flags.append(
                f"Occasional client with {inp.tx_count_24h} versements "
                f"in 24h — potential passeur de fonds (CTAF)")
            tier = max(tier, AlertTier.ALERT)

        # Rule 5: Large amount + new beneficiary
        if (inp.operation_type == "virement"
                and inp.is_new_beneficiary
                and inp.amount >= 5_000):
            reg_flags.append(
                f"Virement of {inp.amount:.2f} DT to new beneficiary")
            tier = max(tier, AlertTier.REVIEW)

        # Rule 6: Duplicate transaction detection
        if inp.has_duplicate_recent:
            reg_flags.append("Potential duplicate transaction detected")
            tier = max(tier, AlertTier.REVIEW)

        # Rule 7: Round amount pattern (potential structuring signal)
        if (inp.is_round_amount
                and inp.amount >= 5_000
                and inp.cumulative_amount_24h >= 15_000):
            reg_flags.append(
                f"Round amount {inp.amount:.0f} DT with "
                f"{inp.cumulative_amount_24h:.0f} DT cumulative in 24h")
            tier = max(tier, AlertTier.REVIEW)

        # Rule 8: Frequency burst (context-aware)
        if (inp.expected_daily_rate > 0
                and inp.tx_count_24h >= inp.expected_daily_rate * 5):
            reg_flags.append(
                f"Frequency burst: {inp.tx_count_24h} txns in 24h vs "
                f"expected {inp.expected_daily_rate:.1f}/day")
            tier = max(tier, AlertTier.REVIEW)

        # ════════════════════════════════════════════════════════
        # LAYER 3: Risk history adjustment
        # ════════════════════════════════════════════════════════

        # Recidivist escalation
        if inp.confirmed_count_30d >= self.RECIDIVIST_CONFIRMED_30D:
            reasons.append(
                f"Recidivist: {inp.confirmed_count_30d} confirmed alerts "
                f"in 30d — escalating")
            tier = min(tier + 1, AlertTier.BLOCK)

        # False-positive dampening (REVIEW → INFO only, no reg flags)
        if (inp.rejected_count_30d >= self.FALSE_POSITIVE_REJECTED_30D
                and tier == AlertTier.REVIEW
                and len(reg_flags) == 0):
            reasons.append(
                f"Dampened: {inp.rejected_count_30d} rejected alerts in 30d, "
                f"no regulatory flags — REVIEW → INFO")
            tier = AlertTier.INFO

        # Cold-start conservatism
        if (inp.maturity_status == "cold"
                and tier == AlertTier.INFO
                and inp.fused_score >= 0.90):
            reasons.append(
                f"Cold-start client (score {inp.fused_score:.4f} >= 0.90) "
                f"— escalating to REVIEW for manual check")
            tier = AlertTier.REVIEW

        # ════════════════════════════════════════════════════════
        # LAYER 4: Assemble explanation
        # ════════════════════════════════════════════════════════
        score_triggered = inp.fused_score >= self.T1
        rule_triggered = len(reg_flags) > 0

        if score_triggered and rule_triggered:
            trigger_type = "both"
        elif rule_triggered:
            trigger_type = "rule"
        elif score_triggered:
            trigger_type = "ml"
        else:
            trigger_type = None

        explanation = None
        if tier > AlertTier.INFO:
            rule_details = [{"rule": f"rule_{i}", "description": desc}
                           for i, desc in enumerate(reg_flags)]

            explanation = assemble_explanation(
                trigger_type=trigger_type,
                ae_explanation=inp.ae_explanation,
                lstm_explanation=inp.lstm_explanation,
                rule_details=rule_details if rule_triggered else None,
            )

        return Decision(
            event_id=inp.event_id,
            tier=tier,
            fused_score=inp.fused_score,
            reasons=reasons,
            regulatory_flags=reg_flags,
            explanation=explanation,
        )