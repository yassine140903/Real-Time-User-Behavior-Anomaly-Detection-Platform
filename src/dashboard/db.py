# src/dashboard/db.py

import os
import psycopg2
import streamlit as st
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

@st.cache_resource
def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "amen_anomaly"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )


def safe_query(query, params=None):
    try:
        conn = get_connection()
        return pd.read_sql(query, conn, params=params)
    except Exception:
        st.cache_resource.clear()
        conn = get_connection()
        return pd.read_sql(query, conn, params=params)


@st.cache_data(ttl=10)
def get_alerts(tier=None, operation_type=None, branch_id=None):
    query = "SELECT * FROM alerts WHERE 1=1"
    params = []

    if tier:
        query += " AND tier = %s"
        params.append(tier)
    if operation_type:
        query += " AND operation_type = %s"
        params.append(operation_type)
    if branch_id:
        query += " AND branch_id = %s"
        params.append(branch_id)

    query += " ORDER BY timestamp DESC"
    return safe_query(query, params)


@st.cache_data(ttl=10)
def get_alert_detail(event_id: str):
    return safe_query(
        "SELECT * FROM alerts WHERE event_id = %s", [event_id]
    )


@st.cache_data(ttl=30)
def get_kpi_data():
    return {
        "by_tier": safe_query(
            "SELECT tier, COUNT(*) as count FROM alerts GROUP BY tier"
        ),
        "by_operation": safe_query(
            "SELECT operation_type, COUNT(*) as count FROM alerts GROUP BY operation_type"
        ),
        "by_branch": safe_query(
            "SELECT branch_id, COUNT(*) as count FROM alerts GROUP BY branch_id ORDER BY count DESC LIMIT 20"
        ),
        "by_day": safe_query(
            "SELECT DATE(timestamp) as day, COUNT(*) as count FROM alerts GROUP BY DATE(timestamp) ORDER BY day"
        ),
    }