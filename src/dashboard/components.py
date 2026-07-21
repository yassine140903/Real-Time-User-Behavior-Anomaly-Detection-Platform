# src/dashboard/components.py

import os
import streamlit as st
import requests
from db import safe_query

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")


def render_sidebar():
    with st.sidebar:
        st.title("🏦 Amen Bank")
        st.caption("Real-Time Anomaly Detection")

        st.divider()

        # Pipeline health
        st.subheader("Pipeline Status")
        try:
            resp = requests.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": "up{job=~'enrichment|scoring|decision'}"},
                timeout=3,
            )
            results = resp.json().get("data", {}).get("result", [])
            all_up = all(r["value"][1] == "1" for r in results) and len(results) > 0
            if all_up:
                st.success(f"● All services running ({len(results)}/3)")
            else:
                up_count = sum(1 for r in results if r["value"][1] == "1")
                st.error(f"● {up_count}/3 services running")
        except Exception:
            st.warning("● Prometheus unreachable")

        st.divider()

        # Pending alerts — column name TBD after we check the schema
        st.subheader("Pending Review")
        try:
            pending = safe_query("""
                SELECT tier, COUNT(*) as count 
                FROM alerts 
                WHERE supervisor_decision = 'pending' 
                GROUP BY tier
            """)
            if not pending.empty:
                for _, row in pending.iterrows():
                    tier = row["tier"]
                    count = row["count"]
                    if tier == "BLOCK":
                        st.error(f"🔴 BLOCK: {count}")
                    elif tier == "ALERT":
                        st.warning(f"🟠 ALERT: {count}")
                    elif tier == "REVIEW":
                        st.info(f"🔵 REVIEW: {count}")
            else:
                st.success("No pending alerts")
        except Exception:
            st.warning("Alerts table unavailable")