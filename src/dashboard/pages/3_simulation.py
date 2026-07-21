# src/dashboard/pages/3_simulation.py

import os
import streamlit as st
import requests
from db import safe_query
from components import render_sidebar
render_sidebar()

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.title("Simulation")
st.caption("Test an operation and get an instant anomaly score")

# ── Load known IDs for dropdowns ────────────────────────
known_clients = safe_query(
    "SELECT DISTINCT client_id::text FROM alerts ORDER BY client_id LIMIT 100"
)
client_list = known_clients["client_id"].tolist() if not known_clients.empty else []

known_employees = safe_query(
    "SELECT DISTINCT employee_id FROM alerts ORDER BY employee_id LIMIT 100"
)
employee_list = known_employees["employee_id"].tolist() if not known_employees.empty else []

# ── Input form ──────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    client_mode = st.radio("Client", ["Select known", "Enter manually"], horizontal=True)
    if client_mode == "Select known" and client_list:
        client_id = st.selectbox("Client ID", client_list)
    else:
        client_id = st.text_input("Client ID")

    employee_mode = st.radio("Employee", ["Select known", "Enter manually"], horizontal=True)
    if employee_mode == "Select known" and employee_list:
        employee_id = st.selectbox("Employee ID", employee_list)
    else:
        employee_id = st.text_input("Employee ID")

    amount = st.number_input("Amount (TND)", min_value=0.0, step=100.0)

with col2:
    operation_type = st.selectbox(
        "Operation Type",
        ["retrait", "versement", "virement", "cheque"]
    )

    beneficiary_id = None
    emitter_id = None

    if operation_type == "virement":
        beneficiary_id = st.text_input("Beneficiary ID")
    elif operation_type == "cheque":
        emitter_id = st.text_input("Emitter ID")

# ── Submit ──────────────────────────────────────────────
if st.button("Score Operation", type="primary"):
    if not client_id or not employee_id:
        st.error("Client ID and Employee ID are required.")
        st.stop()
    if amount <= 0:
        st.error("Amount must be greater than 0.")
        st.stop()
    if operation_type == "virement" and not beneficiary_id:
        st.error("Virement requires a beneficiary ID.")
        st.stop()
    if operation_type == "cheque" and not emitter_id:
        st.error("Cheque requires an emitter ID.")
        st.stop()

    # Build request payload
    payload = {
        "client_id": client_id,
        "employee_id": employee_id,
        "amount": amount,
        "operation_type": operation_type,
    }

    if operation_type == "virement":
        payload["payload"] = {"beneficiary_id": beneficiary_id}
    elif operation_type == "cheque":
        payload["payload"] = {"emitter_id": emitter_id}

    with st.spinner("Scoring..."):
        try:
            resp = requests.post(f"{API_URL}/score", json=payload, timeout=10)

            if not resp.ok:
                st.error(f"API error: {resp.text}")
                st.stop()

            result = resp.json()
        except requests.ConnectionError:
            st.error("Cannot reach the simulation API. Is the service running?")
            st.stop()

    # ── Display result ──────────────────────────────
    st.divider()

    tier = result["tier"]
    score = result["fused_score"]

    tier_colors = {
        "BLOCK": "error", "ALERT": "warning",
        "REVIEW": "info", "INFO": "success"
    }
    getattr(st, tier_colors.get(tier, "info"))(
        f"**Tier: {tier}** — Fused Score: {score:.6f}"
    )

    # Reasons
    if result.get("reasons"):
        st.subheader("Triggered Rules")
        for reason in result["reasons"]:
            st.markdown(f"- {reason}")

    # Regulatory flags
    if result.get("regulatory_flags"):
        st.subheader("Regulatory Flags")
        for flag in result["regulatory_flags"]:
            st.warning(f"⚠️ {flag}")

    # SHAP explanations
    if result.get("ae_explanation"):
        with st.expander("AE Explanation"):
            st.json(result["ae_explanation"])

    if result.get("lstm_explanation"):
        with st.expander("LSTM Explanation"):
            st.json(result["lstm_explanation"])