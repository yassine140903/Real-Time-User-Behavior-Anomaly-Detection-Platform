import os
import streamlit as st
import requests
from db import get_alerts, get_alert_detail

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.title("Alerts")

from components import render_sidebar
render_sidebar()

# ── Filters ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    tier_filter = st.selectbox(
        "Tier", [None, "BLOCK", "ALERT", "REVIEW", "INFO"], 
        format_func=lambda x: "All" if x is None else x
    )
with col2:
    op_filter = st.selectbox(
        "Operation", [None, "retrait", "versement", "virement", "cheque"],
        format_func=lambda x: "All" if x is None else x
    )
with col3:
    status_filter = st.selectbox(
        "Status", ["pending", "confirmed", "rejected", "all"]
    )

# ── Fetch alerts ────────────────────────────────────────
df = get_alerts(tier=tier_filter, operation_type=op_filter)

if status_filter != "all":
    df = df[df["supervisor_decision"] == status_filter]

if df.empty:
    st.info("No alerts match the current filters.")
    st.stop()

st.caption(f"{len(df)} alerts found")

# ── Alert list ──────────────────────────────────────────
for _, row in df.iterrows():
    event_id = str(row["event_id"])
    score = row["fused_score"]
    tier = row["tier"]
    op = row["operation_type"]
    amount = row["amount"]
    ts = row["timestamp"]
    status = row["supervisor_decision"]

    # Color-coded tier badge
    tier_colors = {"BLOCK": "🔴", "ALERT": "🟠", "REVIEW": "🔵", "INFO": "⚪"}
    badge = tier_colors.get(tier, "⚪")

    with st.expander(
        f"{badge} {tier} | {op} | {amount:.2f} TND | {ts} | Score: {score:.4f} | [{status}]"
    ):
        # ── Detail section ──────────────────────────
        detail = get_alert_detail(event_id)
        if detail.empty:
            st.error("Alert detail not found")
            continue
        
        alert = detail.iloc[0]

        # Event info
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Client:** `{alert['client_id']}`")
            st.markdown(f"**Employee:** `{alert['employee_id']}`")
            st.markdown(f"**Branch:** `{alert['branch_id']}`")
        with col_b:
            st.markdown(f"**Score:** `{alert['fused_score']:.6f}`")
            st.markdown(f"**Tier:** `{alert['tier']}`")
            st.markdown(f"**Operation:** `{alert['operation_type']}`")

        # Reasons
        if alert["reasons"]:
            st.subheader("Triggered Rules")
            for reason in alert["reasons"]:
                st.markdown(f"- {reason}")

        # Regulatory flags
        if alert["regulatory_flags"]:
            st.subheader("Regulatory Flags")
            for flag in alert["regulatory_flags"]:
                st.warning(f"⚠️ {flag}")

        # SHAP explanation
        if alert["explanation"]:
            explanation = alert["explanation"]
            
            # Human-readable reasons (NLG output)
            if "reasons" in explanation and explanation["reasons"]:
                st.subheader("Explanation")
                for reason in explanation["reasons"]:
                    st.markdown(f"- {reason}")

            # Raw SHAP details (expandable)
            raw = explanation.get("raw", {})
            
            if raw.get("ae") and raw["ae"].get("top_features"):
                with st.expander("AE Feature Contributions"):
                    st.json(raw["ae"]["top_features"])

            if raw.get("lstm") and raw["lstm"].get("top_deviations"):
                with st.expander("LSTM Deviations"):
                    st.json(raw["lstm"]["top_deviations"])

        # ── Supervisor feedback ─────────────────────
        if alert["supervisor_decision"] == "pending":
            st.divider()
            st.subheader("Supervisor Decision")
            
            comment = st.text_input(
                "Comment (optional)", 
                key=f"comment_{event_id}"
            )
            
            col_confirm, col_reject = st.columns(2)
            with col_confirm:
                if st.button("✅ Confirm", key=f"confirm_{event_id}"):
                    resp = requests.post(f"{API_URL}/feedback", json={
                        "event_id": event_id,
                        "decision": "confirmed",
                        "comment": comment or None,
                    })
                    if resp.ok:
                        st.success("Confirmed")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"Failed: {resp.text}")
            
            with col_reject:
                if st.button("❌ Reject", key=f"reject_{event_id}"):
                    resp = requests.post(f"{API_URL}/feedback", json={
                        "event_id": event_id,
                        "decision": "rejected",
                        "comment": comment or None,
                    })
                    if resp.ok:
                        st.success("Rejected")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"Failed: {resp.text}")
        else:
            st.divider()
            st.markdown(f"**Decision:** {alert['supervisor_decision']} at {alert['decision_timestamp']}")
            if alert["supervisor_notes"]:
                st.markdown(f"**Notes:** {alert['supervisor_notes']}")