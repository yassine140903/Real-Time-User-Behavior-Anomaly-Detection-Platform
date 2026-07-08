-- ============================================
-- FULL RESET: drop everything, rebuild clean
-- ============================================

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
    payload           JSONB,
    is_anomaly        BOOLEAN DEFAULT FALSE,
    anomaly_type      VARCHAR(30)
);

CREATE INDEX idx_tx_client_time ON transactions(client_id, timestamp);
CREATE INDEX idx_tx_employee_time ON transactions(employee_id, timestamp);

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
-- Alerts (populated by Decision Service)
-- ============================================

CREATE TABLE alerts (
    alert_id          UUID PRIMARY KEY,
    event_id          UUID REFERENCES transactions(event_id),
    client_id         UUID REFERENCES clients_master(client_id),
    employee_id       VARCHAR(10) REFERENCES employees_master(employee_id),
    timestamp         TIMESTAMP NOT NULL,
    severity_tier     VARCHAR(10) NOT NULL,
    anomaly_score     NUMERIC(6,4),
    shap_explanation  JSONB,
    supervisor_decision VARCHAR(10) DEFAULT 'pending',
    supervisor_notes  TEXT,
    decision_timestamp TIMESTAMP
);