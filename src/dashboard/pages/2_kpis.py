# src/dashboard/pages/2_kpis.py

import streamlit as st
import plotly.express as px
from db import get_kpi_data
from components import render_sidebar
render_sidebar()

st.title("KPIs")

data = get_kpi_data()

# ── Top-level metrics ───────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

total = data["by_tier"]["count"].sum() if not data["by_tier"].empty else 0

tier_counts = {}
if not data["by_tier"].empty:
    tier_counts = dict(zip(data["by_tier"]["tier"], data["by_tier"]["count"]))

col1.metric("Total Alerts", total)
col2.metric("BLOCK", tier_counts.get("BLOCK", 0))
col3.metric("ALERT", tier_counts.get("ALERT", 0))
col4.metric("REVIEW", tier_counts.get("REVIEW", 0))

st.divider()

# ── Charts ──────────────────────────────────────────────
left, right = st.columns(2)

with left:
    # Alerts by tier
    if not data["by_tier"].empty:
        fig = px.pie(
            data["by_tier"], names="tier", values="count",
            title="Distribution by Tier",
            color="tier",
            color_discrete_map={
                "BLOCK": "#e74c3c",
                "ALERT": "#e67e22",
                "REVIEW": "#3498db",
                "INFO": "#95a5a6",
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    # Alerts by operation type
    if not data["by_operation"].empty:
        fig = px.bar(
            data["by_operation"], x="operation_type", y="count",
            title="Alerts by Operation Type",
            color="operation_type",
        )
        st.plotly_chart(fig, use_container_width=True)

with right:
    # Alerts over time
    if not data["by_day"].empty:
        fig = px.line(
            data["by_day"], x="day", y="count",
            title="Alerts per Day",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Top branches
    if not data["by_branch"].empty:
        fig = px.bar(
            data["by_branch"], x="branch_id", y="count",
            title="Top 20 Branches by Alert Count",
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)