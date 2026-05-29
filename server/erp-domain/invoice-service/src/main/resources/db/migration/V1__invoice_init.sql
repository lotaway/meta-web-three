-- Invoice Service Database Migration V1
-- Created: 2026-05-29
-- Description: Initialize invoice module table

CREATE TABLE IF NOT EXISTS invoice (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    invoice_no VARCHAR(50) NOT NULL UNIQUE COMMENT 'Invoice number',
    order_no VARCHAR(50) COMMENT 'Associated order number',
    customer_id BIGINT NOT NULL COMMENT 'Customer ID',
    customer_name VARCHAR(100) NOT NULL COMMENT 'Customer name',
    customer_tax_no VARCHAR(50) COMMENT 'Customer tax identification number',
    customer_address VARCHAR(200) COMMENT 'Customer address',
    customer_bank VARCHAR(100) COMMENT 'Customer bank name',
    customer_account VARCHAR(50) COMMENT 'Customer bank account',
    type VARCHAR(20) NOT NULL COMMENT 'Invoice type: VAT_SPECIAL, VAT_NORMAL, ELECTRONIC, RECEIPT',
    status VARCHAR(20) NOT NULL DEFAULT 'DRAFT' COMMENT 'Invoice status: DRAFT, PENDING, ISSUED, PRINTED, VOIDED, RED_FLUSHED',
    amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Invoice amount (before tax)',
    tax_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Tax amount',
    total_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Total amount (including tax)',
    tax_rate VARCHAR(10) COMMENT 'Tax rate (e.g., 13%)',
    issue_date TIMESTAMP COMMENT 'Issue date',
    issuer VARCHAR(50) COMMENT 'Issuer username',
    remark VARCHAR(500) COMMENT 'Remarks',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation time',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update time',
    version INT NOT NULL DEFAULT 0 COMMENT 'Optimistic lock version',
    INDEX idx_invoice_no (invoice_no),
    INDEX idx_order_no (order_no),
    INDEX idx_customer_id (customer_id),
    INDEX idx_status (status),
    INDEX idx_type (type),
    INDEX idx_issue_date (issue_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Invoice table';