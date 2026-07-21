# src/dashboard/app.py

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from components import render_sidebar

st.set_page_config(
    page_title="Amen Bank — Anomaly Detection",
    page_icon="🏦",
    layout="wide",
)

st_autorefresh(interval=10_000, key="global_refresh")

render_sidebar()

st.title("Supervisor Dashboard")
st.info("Use the sidebar navigation to access Alerts, KPIs, Simulation, and Health pages.")