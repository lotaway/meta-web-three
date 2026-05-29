-- Settlement Service Database Migration V1
-- Created: 2026-05-29
-- Description: Initialize settlement module tables (settlement_order, reconciliation_record, split_rule)

-- Table: settlement_order
CREATE TABLE IF NOT EXISTS settlement_order (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    settlement_no VARCHAR(50) NOT NULL UNIQUE COMMENT 'Settlement order number',
    order_no VARCHAR(50) NOT NULL COMMENT 'Associated order number',
    merchant_id BIGINT NOT NULL COMMENT 'Merchant ID',
    merchant_name VARCHAR(100) NOT NULL COMMENT 'Merchant name',
    order_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Original order amount',
    settlement_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Settlement amount after commission',
    commission_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Commission amount',
    refund_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Refund amount',
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING' COMMENT 'Settlement status: PENDING, CONFIRMED, PROCESSING, COMPLETED, FAILED, CANCELLED',
    channel VARCHAR(50) COMMENT 'Payment channel',
    settlement_date TIMESTAMP COMMENT 'Settlement date',
    description VARCHAR(500) COMMENT 'Description',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation time',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update time',
    version INT NOT NULL DEFAULT 0 COMMENT 'Optimistic lock version',
    INDEX idx_settlement_no (settlement_no),
    INDEX idx_order_no (order_no),
    INDEX idx_merchant_id (merchant_id),
    INDEX idx_status (status),
    INDEX idx_settlement_date (settlement_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Settlement order table';

-- Table: reconciliation_record
CREATE TABLE IF NOT EXISTS reconciliation_record (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    record_no VARCHAR(50) NOT NULL UNIQUE COMMENT 'Reconciliation record number',
    type VARCHAR(20) NOT NULL COMMENT 'Reconciliation type: DAILY, MONTHLY, CUSTOM',
    reconcile_date TIMESTAMP NOT NULL COMMENT 'Reconciliation date',
    channel VARCHAR(50) NOT NULL COMMENT 'Payment channel',
    total_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Total amount from channel',
    total_count INT NOT NULL DEFAULT 0 COMMENT 'Total transaction count',
    matched_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Matched amount',
    matched_count INT NOT NULL DEFAULT 0 COMMENT 'Matched transaction count',
    unmatched_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Unmatched amount',
    unmatched_count INT NOT NULL DEFAULT 0 COMMENT 'Unmatched transaction count',
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING' COMMENT 'Reconciliation status: PENDING, PROCESSING, COMPLETED, FAILED',
    remark VARCHAR(500) COMMENT 'Remarks',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation time',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update time',
    version INT NOT NULL DEFAULT 0 COMMENT 'Optimistic lock version',
    INDEX idx_record_no (record_no),
    INDEX idx_type (type),
    INDEX idx_channel (channel),
    INDEX idx_reconcile_date (reconcile_date),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Reconciliation record table';

-- Table: split_rule
CREATE TABLE IF NOT EXISTS split_rule (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    rule_no VARCHAR(50) NOT NULL UNIQUE COMMENT 'Split rule number',
    rule_name VARCHAR(100) NOT NULL COMMENT 'Split rule name',
    type VARCHAR(20) NOT NULL COMMENT 'Split type: RATIO, FIXED, MIN_MAX, MIXED',
    merchant_id BIGINT COMMENT 'Merchant ID (null for global rules)',
    ratio DECIMAL(5, 4) COMMENT 'Split ratio (e.g., 0.05 for 5%)',
    fixed_amount DECIMAL(18, 2) COMMENT 'Fixed split amount',
    min_amount DECIMAL(18, 2) COMMENT 'Minimum amount for split',
    max_amount DECIMAL(18, 2) COMMENT 'Maximum amount for split',
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE' COMMENT 'Rule status: ACTIVE, INACTIVE',
    priority INT NOT NULL DEFAULT 0 COMMENT 'Rule priority (higher first)',
    effective_date TIMESTAMP COMMENT 'Effective date',
    expire_date TIMESTAMP COMMENT 'Expiration date',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation time',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update time',
    version INT NOT NULL DEFAULT 0 COMMENT 'Optimistic lock version',
    INDEX idx_rule_no (rule_no),
    INDEX idx_merchant_id (merchant_id),
    INDEX idx_type (type),
    INDEX idx_status (status),
    INDEX idx_priority (priority)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Split rule table';