-- ============================================
-- FULL RESET: drop everything, rebuild clean
-- ============================================

DROP TABLE IF EXISTS evaluation_labels CASCADE;
DROP TABLE IF EXISTS alerts CASCADE;
DROP TABLE IF EXISTS peer_baselines CASCADE;
DROP TABLE IF EXISTS archetype_baselines CASCADE;
DROP TABLE IF EXISTS employee_profile_snapshots CASCADE;
DROP TABLE IF EXISTS profile_snapshots CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS accounts CASCADE;
DROP TABLE IF EXISTS employees_master CASCADE;
DROP TABLE IF EXISTS clients_master CASCADE;

-- ============================================
-- Master tables (no FK dependencies)
-- ============================================

CREATE TABLE clients_master (
    client_id         UUID PRIMARY KEY,
    archetype         VARCHAR(20) NOT NULL,
    account_opening_date DATE NOT NULL,
    nationality       VARCHAR(5) DEFAULT 'TN',
    client_type       VARCHAR(30) NOT NULL,
    home_branch_id    VARCHAR(10) NOT NULL
);

CREATE TABLE employees_master (
    employee_id       VARCHAR(10) PRIMARY KEY,
    branch_id         VARCHAR(10) NOT NULL
);

-- ============================================
-- Dependent tables
-- ============================================

CREATE TABLE accounts (
    account_id        UUID PRIMARY KEY,
    client_id         UUID NOT NULL REFERENCES clients_master(client_id),
    account_type      VARCHAR(20) NOT NULL,
    opening_date      DATE NOT NULL,
    status            VARCHAR(10) DEFAULT 'active'
);

CREATE TABLE transactions (
    event_id          UUID PRIMARY KEY,
    client_id         UUID NOT NULL REFERENCES clients_master(client_id),
    account_id        UUID NOT NULL REFERENCES accounts(account_id),
    employee_id       VARCHAR(10) NOT NULL REFERENCES employees_master(employee_id),
    branch_id         VARCHAR(10) NOT NULL,
    timestamp         TIMESTAMP NOT NULL,
    amount            NUMERIC(12,2) NOT NULL,
    currency          VARCHAR(3) DEFAULT 'TND',
    channel           VARCHAR(10) DEFAULT 'guichet',
    operation_type    VARCHAR(20) NOT NULL,
    payload           JSONB
);

CREATE INDEX idx_tx_client_time ON transactions(client_id, timestamp);
CREATE INDEX idx_tx_employee_time ON transactions(employee_id, timestamp);

-- ============================================
-- Evaluation labels (batch-only, FK safe)
-- ============================================

CREATE TABLE evaluation_labels (
    event_id      UUID PRIMARY KEY REFERENCES transactions(event_id),
    is_anomaly    BOOLEAN NOT NULL,
    anomaly_type  VARCHAR(30)
);

-- ============================================
-- Profile / snapshot tables
-- ============================================

CREATE TABLE profile_snapshots (
    client_id         UUID NOT NULL REFERENCES clients_master(client_id),
    computed_at       TIMESTAMP NOT NULL,
    profile_data      JSONB NOT NULL,
    PRIMARY KEY (client_id, computed_at)
);

CREATE TABLE employee_profile_snapshots (
    employee_id       VARCHAR(10) NOT NULL REFERENCES employees_master(employee_id),
    computed_at       TIMESTAMP NOT NULL,
    profile_data      JSONB NOT NULL,
    PRIMARY KEY (employee_id, computed_at)
);

CREATE TABLE archetype_baselines (
    archetype         VARCHAR(20) PRIMARY KEY,
    baseline_data     JSONB NOT NULL
);

CREATE TABLE peer_baselines (
    branch_id         VARCHAR(10) NOT NULL,
    computed_at       TIMESTAMP NOT NULL,
    baseline_data     JSONB NOT NULL,
    PRIMARY KEY (branch_id, computed_at)
);

-- ============================================
-- Alerts (populated by decisions sink consumer)
--
-- NOTE: No FK to transactions(event_id).
-- In the streaming path, the raw-events sink and decisions sink
-- consume from different topics in parallel. A FK would create
-- a race condition: the decisions sink may arrive before the
-- raw-events sink inserts the transaction row.
-- Referential integrity is guaranteed by pipeline topology —
-- a decision event cannot exist without a corresponding raw event
-- having entered the pipeline.
-- ============================================

CREATE TABLE alerts (
    event_id            UUID PRIMARY KEY,
    client_id           UUID,
    employee_id         VARCHAR(10),
    branch_id           VARCHAR(10),
    timestamp           TIMESTAMP NOT NULL,
    amount              NUMERIC(12,2) NOT NULL,
    operation_type      VARCHAR(20) NOT NULL,
    tier                VARCHAR(10) NOT NULL,
    fused_score         NUMERIC(8,6),
    reasons             JSONB,
    regulatory_flags    JSONB,
    explanation         JSONB,
    supervisor_decision VARCHAR(10) DEFAULT 'pending',
    supervisor_notes    TEXT,
    decision_timestamp  TIMESTAMP,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_alerts_branch ON alerts(branch_id);
CREATE INDEX idx_alerts_tier ON alerts(tier);
CREATE INDEX idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX idx_alerts_operation ON alerts(operation_type);