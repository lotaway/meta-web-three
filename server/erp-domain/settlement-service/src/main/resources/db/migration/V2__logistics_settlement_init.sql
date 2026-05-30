-- Settlement Service Database Migration V2
-- Created: 2026-05-30
-- Description: Add logistics settlement table for automatic freight settlement

-- Table: logistics_settlement
CREATE TABLE IF NOT EXISTS logistics_settlement (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    settlement_no VARCHAR(50) NOT NULL UNIQUE COMMENT 'Settlement order number',
    tracking_no VARCHAR(50) NOT NULL COMMENT 'Logistics tracking number',
    order_no VARCHAR(50) COMMENT 'Associated order number',
    carrier_id BIGINT NOT NULL COMMENT 'Carrier ID',
    carrier_name VARCHAR(100) NOT NULL COMMENT 'Carrier name',
    freight DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Freight amount',
    handling_fee DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Handling fee',
    discount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Discount',
    total_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Total settlement amount',
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING' COMMENT 'Settlement status: PENDING, CONFIRMED, PROCESSING, COMPLETED, FAILED, CANCELLED',
    billing_cycle VARCHAR(20) NOT NULL DEFAULT 'MONTHLY' COMMENT 'Billing cycle: DAILY, WEEKLY, MONTHLY',
    settlement_date TIMESTAMP COMMENT 'Settlement date',
    paid_at TIMESTAMP COMMENT 'Payment date',
    remark VARCHAR(500) COMMENT 'Remarks',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation time',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update time',
    version INT NOT NULL DEFAULT 0 COMMENT 'Optimistic lock version',
    deleted TINYINT NOT NULL DEFAULT 0 COMMENT 'Soft delete flag',
    INDEX idx_settlement_no (settlement_no),
    INDEX idx_tracking_no (tracking_no),
    INDEX idx_order_no (order_no),
    INDEX idx_carrier_id (carrier_id),
    INDEX idx_carrier_name (carrier_name),
    INDEX idx_status (status),
    INDEX idx_billing_cycle (billing_cycle),
    INDEX idx_settlement_date (settlement_date),
    INDEX idx_paid_at (paid_at),
    UNIQUE KEY uk_tracking_no (tracking_no, deleted)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Logistics freight settlement table';