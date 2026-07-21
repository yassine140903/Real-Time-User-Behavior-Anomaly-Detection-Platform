# src/dashboard/pages/4_health.py

import os
import streamlit as st
import requests
from components import render_sidebar
render_sidebar()

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

SERVICES = {
    "enrichment": "Enrichment Service",
    "scoring": "Scoring Service",
    "decision": "Decision Service",
    "raw-events-sink": "Raw Events Sink",
    "decisions-sink": "Decisions Sink",
    "kafka-exporter": "Kafka Exporter",
}

st.title("Pipeline Health")


def prom_query(query):
    """Query Prometheus and return results list."""
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=3,
        )
        return resp.json().get("data", {}).get("result", [])
    except Exception:
        return None


# ── Service status ──────────────────────────────────────
st.header("Service Status")

results = prom_query("up")

if results is None:
    st.error("Prometheus is unreachable.")
    st.stop()

# Build a lookup: job_name -> up/down
status_map = {}
for r in results:
    job = r["metric"].get("job", "unknown")
    value = r["value"][1]
    status_map[job] = value == "1"

cols = st.columns(3)
for i, (job, label) in enumerate(SERVICES.items()):
    with cols[i % 3]:
        if job not in status_map:
            st.warning(f"⚪ {label} — Not found")
        elif status_map[job]:
            st.success(f"🟢 {label}")
        else:
            st.error(f"🔴 {label} — DOWN")

st.divider()

# ── Throughput ──────────────────────────────────────────
st.header("Throughput (events/min)")

throughput_jobs = ["enrichment", "scoring", "decision", "raw-events-sink", "decisions-sink"]

cols = st.columns(len(throughput_jobs))
for i, job in enumerate(throughput_jobs):
    with cols[i]:
        results = prom_query(
            f'rate(events_processed_total{{job="{job}"}}[1m])'
        )
        if results:
            # Sum across all operation_type labels
            total_rate = sum(float(r["value"][1]) for r in results)
            st.metric(SERVICES[job], f"{total_rate * 60:.1f}")
        else:
            st.metric(SERVICES[job], "—")

st.divider()

# ── Processing latency ─────────────────────────────────
st.header("Processing Latency (p95, ms)")

latency_jobs = ["enrichment", "scoring", "decision"]

cols = st.columns(len(latency_jobs))
for i, job in enumerate(latency_jobs):
    with cols[i]:
        results = prom_query(
            f'histogram_quantile(0.95, rate(processing_duration_seconds_bucket{{job="{job}"}}[5m]))'
        )
        if results and results[0]["value"][1] != "NaN":
            latency_ms = float(results[0]["value"][1]) * 1000
            st.metric(SERVICES[job], f"{latency_ms:.1f} ms")
        else:
            st.metric(SERVICES[job], "—")

st.divider()

# ── Errors ──────────────────────────────────────────────
st.header("Errors (last 5 min)")

error_jobs = ["enrichment", "scoring", "decision", "raw-events-sink", "decisions-sink"]

has_errors = False
for job in error_jobs:
    results = prom_query(
        f'increase(errors_total{{job="{job}"}}[5m])'
    )
    if results:
        for r in results:
            count = float(r["value"][1])
            if count > 0:
                error_class = r["metric"].get("error_class", "unknown")
                st.error(f"**{SERVICES[job]}** — {error_class}: {int(count)} errors")
                has_errors = True

if not has_errors:
    st.success("No errors in the last 5 minutes.")

st.divider()

# ── Consumer lag ────────────────────────────────────────
st.header("Consumer Lag")

lag_results = prom_query("kafka_consumergroup_lag")

if lag_results:
    lag_data = []
    for r in lag_results:
        group = r["metric"].get("consumergroup", "unknown")
        topic = r["metric"].get("topic", "unknown")
        partition = r["metric"].get("partition", "?")
        lag = int(float(r["value"][1]))
        lag_data.append({
            "Consumer Group": group,
            "Topic": topic,
            "Partition": partition,
            "Lag": lag,
        })

    if lag_data:
        # Highlight high lag
        total_lag = sum(d["Lag"] for d in lag_data)
        if total_lag > 100:
            st.warning(f"Total consumer lag: {total_lag} messages")
        else:
            st.success(f"Total consumer lag: {total_lag} messages")

        st.dataframe(lag_data, use_container_width=True)
else:
    st.info("No consumer lag data available.")