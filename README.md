# Real-Time User Behavior Anomaly Detection Platform

A real-time fraud and behavioral anomaly detection platform built for Amen Bank branch (guichet) operations, covering the four core teller transaction types: retrait (withdrawal), versement (deposit), virement (transfer), and remise de chèques (cheque deposit). The platform is designed for branch supervisors and the AML/compliance function: it scores every transaction as it happens, fuses machine-learning anomaly signals with hard regulatory rules, and surfaces a ranked, explainable alert queue instead of a wall of raw transaction logs. It targets three overlapping risk categories — classic transaction fraud, AML structuring/smurfing patterns defined by Tunisian banking regulation (BCT, CTAF, Law n°41-2024), and softer behavioral drift in how a client or employee normally operates — and treats client and employee as two actors whose behavior is profiled and scored independently.

## Architecture Overview

Transactions move through a four-stage streaming pipeline built on Kafka-compatible topics: `raw-events` → `enriched-events` → `scored-events` → `decisions`. A simulator (or a real core-banking feed, in production) publishes raw teller events to `raw-events`. The enrichment service is the only component allowed to read and write behavioral state — it pulls the client and employee's "hot" profile from Redis, computes contextual features (z-scores against archetype baselines, rolling counts, cumulative amounts, near-threshold flags, duplicate detection) and republishes an enriched event. The scoring service consumes enriched events and fuses two models: an autoencoder (AE) that scores single-event reconstruction error, and an LSTM sequence model that scores deviation from the client's recent transaction history. Because a brand-new client has no sequence to compare against, the two scores are combined with a sigmoid-weighted average keyed on account age — young accounts lean almost entirely on the AE, mature accounts lean increasingly on the LSTM — rather than a fixed weight. Each score is also converted to a percentile against a reference distribution so AE and LSTM, which live on different raw scales, become comparable before fusion. SHAP values are computed alongside the fused score so every alert ships with a ranked list of the features that drove it. The decision service is the last, stateless stage: it maps the fused score to an alert tier, then layers deterministic AML/regulatory rules and risk-history adjustments on top, and only ever escalates — rules can push a tier up, never suppress a genuine ML-driven signal down. Four tiers come out the other end: INFO (no action), REVIEW (queued for a supervisor), ALERT (elevated risk, prioritized in the queue), and BLOCK (hard stop, immediate escalation). Two sink consumers persist the pipeline's output — raw events into `transactions`, decisions into `alerts` — independently and in parallel, and profile state is kept "hot" in Redis for millisecond-latency reads while the durable "cold" copy lives in PostgreSQL, refreshed nightly by batch jobs.

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Streaming | Redpanda (Kafka API), kafka-python |
| ML | PyTorch (Autoencoder + LSTM), scikit-learn, SHAP |
| Feature Store | Redis (hot profiles, buffers, archetype baselines) |
| Database | PostgreSQL 17 (cold profiles, transactions, alerts, evaluation labels) |
| Orchestration | Apache Airflow (LocalExecutor, DockerOperator) |
| MLOps | MLflow (tracking, model registry) |
| Monitoring | Prometheus, Alertmanager, kafka-exporter |
| Dashboard | Streamlit, Plotly |
| API | FastAPI, Uvicorn |
| Containerization | Docker Compose (20 services) |

## Project Structure

```
.
├── docker-compose.yml          # 20-service orchestration (infra, pipeline, MLOps, dashboard)
├── Dockerfile.api               # streaming services, API, batch jobs, simulator
├── Dockerfile.airflow            # Airflow webserver/scheduler
├── Dockerfile.dashboard          # Streamlit supervisor dashboard
├── Dockerfile.webhook            # Prometheus→Airflow drift-retrain relay
├── config/
│   ├── archetype.yaml             # 5 client archetypes (population mix, op mix, amount/frequency priors)
│   ├── init.sql                   # PostgreSQL schema (clients, transactions, alerts, snapshots)
│   ├── prometheus.yml / alert_rules.yml   # scrape config + drift/health alert rules
│   └── alertmanager.yml           # routes drift alerts to the webhook relay
├── dags/
│   ├── nightly_profiles.py        # batch profile/baseline refresh → hydrate Redis
│   ├── weekly_retrain.py          # scheduled retrain (also triggered by drift/feedback)
│   └── feedback_monitor.py        # daily supervisor-rejection-rate check
├── src/
│   ├── generator/                 # synthetic client population + anomaly injection
│   ├── streaming/                 # simulator, enrichment/scoring/decision services, sinks, DLQ
│   ├── enrichment/                # EnrichmentCore — feature computation (pure, no I/O side effects)
│   ├── scoring/                   # ScoreFusion (AE + LSTM), SHAP explanation core
│   ├── decision/                  # DecisionService — tiering, regulatory rules, explainer
│   ├── training/                  # model definitions + train_all entrypoint (MLflow-tracked)
│   ├── batch/                     # nightly profile/baseline/employee-profile jobs, hydrate_redis
│   ├── dashboard/                 # Streamlit app (Alerts, KPIs, Simulation, Health pages)
│   ├── api/                       # FastAPI simulation-control API
│   ├── webhook/                   # Alertmanager → Airflow retrain relay
│   └── monitoring/                # Prometheus metrics definitions
├── models/                      # trained artifacts (autoencoder.pt, lstm.pt, scaler, reference scores)
├── scripts/                     # one-off/maintenance scripts (seeding, calibration, sanity checks)
└── tests/                       # decision, fusion, and SHAP end-to-end tests
```

## Services

| Service | Role | Port |
|---|---|---|
| **Infrastructure** | | |
| redpanda | Kafka-compatible event broker | 9092, 8082, 9644 |
| redis | Hot profile/buffer/baseline store | 6379 |
| postgres | Cold storage: clients, transactions, alerts, snapshots | 5432 |
| **Streaming Pipeline** | | |
| simulator | Publishes synthetic teller events to `raw-events` | — |
| enrichment-service | Computes contextual features, hot-profile lookups | 9100 |
| scoring-service | AE + LSTM fusion, SHAP explanation | 9101 |
| decision-service | Tiering, regulatory rules, alert assembly | 9102 |
| **Sink Consumers** | | |
| raw-events-sink | Persists raw events to `transactions` | 9103 |
| decisions-sink | Persists decisions to `alerts`, updates Redis risk history | 9104 |
| **Batch / Orchestration** | | |
| hydrate-redis | Loads cold profiles/baselines from Postgres into Redis | — |
| db-seed | Loads synthetic CSV data into Postgres | — |
| airflow-init | Airflow DB migration + admin user bootstrap | — |
| airflow-webserver | Airflow UI | 8080 |
| airflow-scheduler | Runs `nightly_profiles`, `weekly_retrain`, `feedback_monitor` DAGs | — |
| **MLOps** | | |
| training | Trains AE + LSTM, logs runs/artifacts to MLflow | — |
| mlflow | Experiment tracking + model registry UI | 5001 |
| **Monitoring** | | |
| prometheus | Metrics scraping (pipeline latency, throughput, errors) | 9090 |
| alertmanager | Routes drift/health alerts | 9093 |
| kafka-exporter | Redpanda/Kafka metrics for Prometheus | 9308 |
| webhook-relay | Forwards drift alerts into an Airflow retrain trigger | — |
| **User-Facing** | | |
| dashboard | Streamlit supervisor dashboard | 8501 |
| simulation-api | FastAPI control plane for the simulator | 8000 |

## Getting Started

**Prerequisites:** Docker and Docker Compose.

```bash
git clone <repo-url>
cd Real-Time-User-Behavior-Anomaly-Detection-Platform

# 1. Generate synthetic clients, transactions, and labeled anomalies
python -m src.generator.main

# 2. Seed PostgreSQL from the generated CSVs
docker compose --profile seed up db-seed

# 3. Build all images
docker compose build

# 4. Bring up the full platform
docker compose up -d
```

For a live demo of the streaming pipeline rather than a full historical backfill, hydrate Redis without pre-loading rolling buffers and start the simulator from a fixed recent date:

```bash
docker compose run --rm hydrate-redis python -m src.batch.hydrate_redis --skip-buffers
docker compose run --rm simulator python -m src.streaming.simulator --start_date 2025-06-20
```

This lets the buffers build up organically as events stream through, which is more representative of what a supervisor watching the dashboard would actually see. The dashboard is then available at `http://localhost:8501`.

## Key Design Decisions

**Single Kafka topic per stage, partitioned by client_id.** All four operation types flow through one topic at each pipeline stage rather than one topic per operation type. This preserves cross-operation temporal ordering within each partition — critical for the LSTM sequence model, which needs to see a client's retrait followed by their versement in the correct order. The client_id partition key guarantees per-client ordering while still allowing horizontal consumer scaling.

**EnrichmentCore has no I/O side effects.** All Redis/Postgres access happens in the wrapping service; `EnrichmentCore.enrich_event()` takes a profile and buffers already fetched and returns a pure feature dict. This makes feature computation independently testable and reusable between the streaming service and the batch backfill path used for training data.

**Rules only escalate, never de-escalate.** The decision service's four layers (score tiering → regulatory rules → risk-history adjustment → explanation) can only raise a tier once assigned. A genuine ML-driven signal can never be silently downgraded by a rule.

**No foreign key from `alerts` to `transactions`.** This is intentional, not an oversight: the raw-events sink and decisions sink consume from different topics in parallel, so a decision can legitimately reach Postgres before its corresponding raw event does. Referential integrity is guaranteed by pipeline topology instead of a database constraint.

**Three-tier cold-start scoring.** Clients with fewer than 5 events in their sequence buffer receive a neutral fused score of 0.5 with cold_start=True — the models are not invoked at all. The decision service still runs all regulatory rules (a structuring pattern is suspicious even from a new client) but caps the final tier at REVIEW maximum. Between 5 and 9 events, only the AE scores (LSTM weight forced to 0). At 10+ events, full AE+LSTM fusion engages with sigmoid weighting by account maturity.

**DLQ with failure classification.** Every streaming consumer wraps `process_fn` in a resilient handler that distinguishes transient infrastructure errors (Redis/Postgres connection issues — retried with exponential backoff) from poison-pill events (malformed payloads — sent straight to a `dlq` topic with the full stack trace and original event, no retry wasted).

**Three independent retrain triggers.** Models retrain on a weekly cron schedule, on statistical drift detected by Prometheus/Alertmanager (routed through a webhook relay into an Airflow DAG run), and on supervisor feedback divergence (rejection rate on ALERT/BLOCK exceeding 50% over 7 days, minimum 10 reviews) — so the model can react to both slow drift and sudden disagreement with human reviewers, not just the calendar.

## MLOps Pipeline

Retraining is triggered three ways: a weekly Sunday cron (`weekly_retrain` DAG), a Prometheus-detected drift alert forwarded through the webhook relay to Airflow, and a daily feedback-divergence check that fires when supervisors are rejecting more than half of ALERT/BLOCK alerts. Every trigger runs the same DAG — batch re-enrichment, a feature-completeness/NaN-rate validation gate, then a containerized training run (`train_all`) that logs metrics, parameters, and artifacts to MLflow. Promotion is automatic: the training script compares the new AUC against models/production_metrics.json and overwrites the production model files only if the new AUC meets or exceeds the current value. MLflow's role is post-mortem — when a promotion gate rejects a model at 4 AM, engineers can inspect the full run on Monday.

## Dashboard

The Streamlit supervisor dashboard at `http://localhost:8501` auto-refreshes every 10 seconds and is organized into four pages: **Alerts** (the live, tiered queue with SHAP-backed explanations and regulatory flags), **KPIs** (branch and employee-level volume, alert-rate, and tier-distribution views), **Simulation** (control surface for the event simulator), and **Health** (pipeline latency, throughput, and DLQ status pulled from Prometheus). Supervisors action alerts directly from the Alerts page — confirming or rejecting them writes back to `alerts.supervisor_decision`, which feeds the feedback-divergence retrain trigger, closing the loop between human review and model retraining.

## Anomaly Scenarios

The synthetic data generator injects thirteen labeled anomaly scenarios across three difficulty tiers, used both to validate detection and to drive the live simulation demo.

| Scenario | Difficulty | Actor | Description |
|---|---|---|---|
| amount_spike | Easy | Client | Single transaction far above the client's usual amount |
| round_amount | Easy | Client | Suspiciously round transaction amount (5000, 10000, 15000, 20000 DT) |
| duplicate_virement | Easy | Client | Near-identical transfer repeated in a short window |
| volume_spike | Easy | Employee | Sudden spike in an employee's processed transaction volume |
| frequency_burst | Medium | Client | Sudden spike in transaction count over the client's expected rate |
| cumulative_threshold | Medium | Client | Multiple transactions cumulatively approaching a regulatory threshold |
| timing_anomaly | Medium | Client | Transaction outside normal branch hours |
| client_concentration | Medium | Employee | Unusually concentrated client-employee pairing |
| smurfing | Hard | Client | Sustained structuring of deposits just under the 10,000 DT reporting threshold |
| cheque_structuring | Hard | Client | Cheque amounts kept just under the 30,000 DT legal ceiling (Law n°41-2024) |
| money_mule | Hard | Client | Account used as pass-through for rapid in/out transfers |
| behavioral_drift | Hard | Client | Gradual shift in transaction patterns over weeks |
| cross_actor | Hard | Cross | Coordinated anomalous behavior between client and employee |

## Model Performance

| Model | AUC | Note |
|---|---|---|
| Autoencoder | 0.9714 | Single-event reconstruction error |
| LSTM | 0.9355 | Sequence-deviation score, mature clients only |

These figures come from training and validation on synthetic data generated from the archetype-based simulator, not production transaction history. They demonstrate that the fusion architecture and feature design are capable of separating injected anomalies from normal behavior — they are not a claim about real-world detection performance, which would need to be established against labeled production data.

## Production Roadmap

Moving this from a Docker Compose demo to production would require: Kubernetes for orchestration and autoscaling in place of Compose; a CI/CD pipeline for automated testing, image builds, and gated model promotion; a real secrets manager (Vault or a cloud KMS) instead of `.env`-file credentials; load testing to validate throughput under realistic branch-network transaction volume; a defined data retention/archival policy for transactions, alerts, and profile snapshots; and centralized log aggregation (ELK or Loki) in place of per-container logs.

---

Built as part of an internship at Amen Bank, DCSI.
