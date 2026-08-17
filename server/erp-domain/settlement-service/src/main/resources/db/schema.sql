-- Settlement Service Database Schema
-- Initialize settlement module tables (settlement_order, reconciliation_record, split_rule)
-- Includes: logistics_settlement for automatic freight settlement

-- Table: settlement_order
CREATE TABLE IF NOT EXISTS settlement_order (
    id BIGSERIAL PRIMARY KEY,
    settlement_no VARCHAR(50) NOT NULL UNIQUE,
    order_no VARCHAR(50) NOT NULL,
    merchant_id BIGINT NOT NULL,
    merchant_name VARCHAR(100) NOT NULL,
    order_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    settlement_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    commission_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    refund_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    channel VARCHAR(50),
    settlement_date TIMESTAMP,
    description VARCHAR(500),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_settlement_order_settlement_no ON settlement_order (settlement_no);
CREATE INDEX IF NOT EXISTS idx_settlement_order_no ON settlement_order (order_no);
CREATE INDEX IF NOT EXISTS idx_settlement_order_merchant_id ON settlement_order (merchant_id);
CREATE INDEX IF NOT EXISTS idx_settlement_order_status ON settlement_order (status);
CREATE INDEX IF NOT EXISTS idx_settlement_order_date ON settlement_order (settlement_date);

-- Table: reconciliation_record
CREATE TABLE IF NOT EXISTS reconciliation_record (
    id BIGSERIAL PRIMARY KEY,
    record_no VARCHAR(50) NOT NULL UNIQUE,
    type VARCHAR(20) NOT NULL,
    reconcile_date TIMESTAMP NOT NULL,
    channel VARCHAR(50) NOT NULL,
    total_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    total_count INT NOT NULL DEFAULT 0,
    matched_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    matched_count INT NOT NULL DEFAULT 0,
    unmatched_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    unmatched_count INT NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    remark VARCHAR(500),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_reconciliation_record_no ON reconciliation_record (record_no);
CREATE INDEX IF NOT EXISTS idx_reconciliation_type ON reconciliation_record (type);
CREATE INDEX IF NOT EXISTS idx_reconciliation_channel ON reconciliation_record (channel);
CREATE INDEX IF NOT EXISTS idx_reconciliation_date ON reconciliation_record (reconcile_date);
CREATE INDEX IF NOT EXISTS idx_reconciliation_status ON reconciliation_record (status);

-- Table: split_rule
CREATE TABLE IF NOT EXISTS split_rule (
    id BIGSERIAL PRIMARY KEY,
    rule_no VARCHAR(50) NOT NULL UNIQUE,
    rule_name VARCHAR(100) NOT NULL,
    type VARCHAR(20) NOT NULL,
    merchant_id BIGINT,
    ratio DECIMAL(5, 4),
    fixed_amount DECIMAL(18, 2),
    min_amount DECIMAL(18, 2),
    max_amount DECIMAL(18, 2),
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    priority INT NOT NULL DEFAULT 0,
    effective_date TIMESTAMP,
    expire_date TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_split_rule_no ON split_rule (rule_no);
CREATE INDEX IF NOT EXISTS idx_split_rule_merchant_id ON split_rule (merchant_id);
CREATE INDEX IF NOT EXISTS idx_split_rule_type ON split_rule (type);
CREATE INDEX IF NOT EXISTS idx_split_rule_status ON split_rule (status);
CREATE INDEX IF NOT EXISTS idx_split_rule_priority ON split_rule (priority);

-- Table: logistics_settlement
CREATE TABLE IF NOT EXISTS logistics_settlement (
    id BIGSERIAL PRIMARY KEY,
    settlement_no VARCHAR(50) NOT NULL UNIQUE,
    tracking_no VARCHAR(50) NOT NULL,
    order_no VARCHAR(50),
    carrier_id BIGINT NOT NULL,
    carrier_name VARCHAR(100) NOT NULL,
    freight DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    handling_fee DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    discount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    total_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    billing_cycle VARCHAR(20) NOT NULL DEFAULT 'MONTHLY',
    settlement_date TIMESTAMP,
    paid_at TIMESTAMP,
    remark VARCHAR(500),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INT NOT NULL DEFAULT 0,
    deleted SMALLINT NOT NULL DEFAULT 0,
    UNIQUE (tracking_no, deleted)
);
CREATE INDEX IF NOT EXISTS idx_logistics_settlement_no ON logistics_settlement (settlement_no);
CREATE INDEX IF NOT EXISTS idx_logistics_tracking_no ON logistics_settlement (tracking_no);
CREATE INDEX IF NOT EXISTS idx_logistics_order_no ON logistics_settlement (order_no);
CREATE INDEX IF NOT EXISTS idx_logistics_carrier_id ON logistics_settlement (carrier_id);
CREATE INDEX IF NOT EXISTS idx_logistics_carrier_name ON logistics_settlement (carrier_name);
CREATE INDEX IF NOT EXISTS idx_logistics_status ON logistics_settlement (status);
CREATE INDEX IF NOT EXISTS idx_logistics_billing_cycle ON logistics_settlement (billing_cycle);
CREATE INDEX IF NOT EXISTS idx_logistics_settlement_date ON logistics_settlement (settlement_date);
CREATE INDEX IF NOT EXISTS idx_logistics_paid_at ON logistics_settlement (paid_at);