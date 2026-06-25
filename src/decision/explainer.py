"""
Explanation templates and assembler.
"""

AE_TEMPLATES = {
    "amount": "Transaction amount is {value} DT",
    "amount_to_mean_ratio": "Amount is {value}x the client's usual average",
    "hour": "Transaction occurred at {value}h",
    "op_type_probability": "This operation type has only {value}% probability for this client",
    "day_distance_from_preferred": "Transaction is {value} days from client's usual payday",
    "tx_count_24h": "Client made {value} transactions in the last 24h",
    "tx_count_7d": "Client made {value} transactions in the last 7 days",
    "cumulative_amount_24h": "Cumulative amount in 24h reached {value} DT",
    "near_threshold_count_7d": "{value} near-threshold transactions in the last 7 days",
    "employee_tx_count_24h": "Employee processed {value} transactions in 24h",
    "same_employee_client_count_24h": "Same employee-client pair: {value} transactions in 24h",
    "is_new_counterparty": "Counterparty is new (never seen before)",
    "is_round_amount": "Amount is a suspicious round number",
    "is_above_threshold": "Amount exceeds the regulatory threshold",
    "is_near_threshold": "Amount is just below the regulatory threshold",
    "is_near_cheque_ceiling": "Cheque amount is near the 30,000 DT legal ceiling",
    "is_home_branch": "Transaction occurred at a different branch than usual",
    "is_outside_hours": "Transaction is outside normal business hours",
    "has_duplicate_recent": "A similar transaction was made recently",
    "account_age_days": "Account is only {value} days old",
}

LSTM_TEMPLATES = {
    "amount": "Expected amount ~{expected} DT, observed {actual} DT",
    "amount_to_mean_ratio": "Expected ratio ~{expected}x, observed {actual}x",
    "hour": "Expected transaction around {expected}h, occurred at {actual}h",
    "op_type_probability": "Expected op probability ~{expected}%, observed {actual}%",
    "day_distance_from_preferred": "Expected {expected} days from payday, observed {actual} days",
    "tx_count_24h": "Expected ~{expected} transactions in 24h, observed {actual}",
    "tx_count_7d": "Expected ~{expected} transactions in 7d, observed {actual}",
    "cumulative_amount_24h": "Expected cumulative ~{expected} DT, observed {actual} DT",
    "near_threshold_count_7d": "Expected ~{expected} near-threshold ops in 7d, observed {actual}",
    "employee_tx_count_24h": "Expected ~{expected} employee ops in 24h, observed {actual}",
    "same_employee_client_count_24h": "Expected ~{expected} same-pair ops in 24h, observed {actual}",
}


def assemble_explanation(trigger_type, ae_explanation=None,
                         lstm_explanation=None, rule_details=None):
    reasons = []

    if trigger_type in ("ml", "both"):
        if ae_explanation and ae_explanation.get("top_features"):
            for feature, (shap_val, feat_val) in ae_explanation["top_features"].items():
                template = AE_TEMPLATES.get(feature)
                if template:
                    if "{value}" in template:
                        reasons.append(template.format(value=round(feat_val, 2)))
                    else:
                        reasons.append(template)

        if lstm_explanation and lstm_explanation.get("top_deviations"):
            for feature, (expected, actual) in lstm_explanation["top_deviations"].items():
                template = LSTM_TEMPLATES.get(feature)
                if template:
                    reasons.append(template.format(
                        expected=round(expected, 2),
                        actual=round(actual, 2)
                    ))

    if trigger_type in ("rule", "both"):
        if rule_details:
            for rule in rule_details:
                reasons.append(rule["description"])

    return {
        "trigger": trigger_type,
        "reasons": reasons,
        "raw": {
            "ae": ae_explanation,
            "lstm": lstm_explanation,
            "rules": rule_details,
        }
    }